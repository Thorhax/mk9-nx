// Offline, fail-closed DECL repair. Does not modify sequencer code or packages.
// Usage: repair_vertex_variant_declarations template.bin actual.bin fresh-output.bin
#include "../../app/src/me/me_shader_identity.h"
#include "../../app/src/me/me_vertex_fetch_selection.h"

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
using Bytes = std::vector<uint8_t>;
constexpr size_t kCap = 16u << 20;
[[noreturn]] void Reject(const std::string& reason) { throw std::runtime_error(reason); }
Bytes Read(const char* path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f || f.tellg() <= 0 || uint64_t(f.tellg()) > kCap) Reject("invalid input size");
  Bytes b(size_t(f.tellg())); f.seekg(0);
  if (!f.read(reinterpret_cast<char*>(b.data()), b.size())) Reject("input read failed");
  return b;
}
uint32_t Be(const Bytes& b, size_t p) {
  if (p > b.size() || b.size() - p < 4) Reject("container word out of bounds");
  return uint32_t(b[p]) << 24 | uint32_t(b[p+1]) << 16 | uint32_t(b[p+2]) << 8 | b[p+3];
}
void Put(Bytes& b, size_t p, uint32_t v) {
  for (unsigned i = 0; i < 4; ++i) b.at(p+i) = uint8_t(v >> (24-i*8));
}
struct Container {
  Bytes bytes;
  size_t start, size, decl;
  std::vector<uint32_t> words, entries;
  explicit Container(const char* path) : bytes(Read(path)) {
    if (bytes.size() < 28 || (Be(bytes,0) & 0xFFFFFF00u) != 0x102A1100u || !(Be(bytes,0)&1))
      Reject("only 2008 vertex containers supported");
    const size_t virtual_size = Be(bytes,4), h = Be(bytes,24);
    if (uint64_t(virtual_size)+Be(bytes,8)!=bytes.size() || h > virtual_size || virtual_size-h < 40)
      Reject("invalid virtual shader header");
    start = virtual_size + uint64_t(Be(bytes,h)); size = Be(bytes,h+4);
    if (!size || size%12 || start > bytes.size() || size > bytes.size()-start)
      Reject("invalid sequencer extent");
    const uint32_t previous = Be(bytes,h+24), count = Be(bytes,h+28);
    decl = h+36+uint64_t(previous)*4;
    if (!count || count > 64 || previous > 1024 || decl > virtual_size ||
        uint64_t(count)*4 > virtual_size-decl) Reject("invalid or empty DECL");
    for (size_t i=0;i<size/4;++i) words.push_back(Be(bytes,start+i*4));
    for (size_t i=0;i<count;++i) entries.push_back(Be(bytes,decl+i*4));
  }
};
struct Location { unsigned clause; unsigned cluster; unsigned clause_first; unsigned clause_count;
                  unsigned opcode; bool serialized; };
std::map<unsigned,Location> FetchLocations(const std::vector<uint32_t>& w) {
  std::map<unsigned,Location> result;
  size_t end=w.size()/3;
  unsigned cluster=~0u;
  for (size_t pair=0;pair<end;++pair) {
    const uint64_t cf[2]{uint64_t(w[pair*3]) | uint64_t(w[pair*3+1]&65535)<<32,
                         uint64_t(w[pair*3+1]>>16) | uint64_t(w[pair*3+2])<<16};
    for (unsigned half=0;half<2;++half) {
      const uint64_t c=cf[half]; const unsigned op=(c>>44)&15;
      if (!((op>=1 && op<=6) || op==13 || op==14)) continue;
      const unsigned first=c&4095, count=(c>>12)&7, seq=(c>>16)&4095;
      if (first) end=std::min(end,size_t(first));
      if (!count || uint64_t(first)+count>w.size()/3) Reject("invalid EXEC extent");
      // A conditional EXEC can enter without its preceding full-fetch base.
      for (unsigned i=0;i<count;++i) {
        const unsigned ins=first+i;
        if (!(seq>>(i*2)&1)) continue;
        if (op!=1 && op!=2) Reject("conditional fetch EXEC unsupported for base continuity proof");
        const unsigned p=ins*3;
        if ((w[p]&31)!=0) Reject("non-vertex fetch in VS fetch-only proof");
        if (!(w[p+1] & 0x40000000u)) cluster=ins;
        if (cluster==~0u) Reject("mini fetch without full base");
        if (!result.emplace(ins,Location{unsigned(pair*2+half),cluster,first,count,op,
                                        bool(seq>>(i*2+1)&1)}).second)
          Reject("multiply executed fetch unsupported");
      }
    }
  }
  return result;
}
bool UnconditionalFetchOnlyInterval(unsigned low,unsigned high,
                                    const std::map<unsigned,Location>& locations,
                                    const std::vector<uint32_t>& words) {
  if(high<low || high-low>=16) return false;
  for(unsigned i=low;i<=high;++i) {
    const auto it=locations.find(i);
    if(it==locations.end() || it->second.opcode!=1 || it->second.serialized ||
       (words[i*3+1]&0x80000000u)) return false;
  }
  return true;
}
// SDK ParseVertexFetchInstruction inherits only index source, fetch constant,
// stride and index rounding. Format/offset belong to EACH operation. Full-base
// equivalence is independent of particular runtime buffer addresses when the
// constant selector is the SAME. No ALU, serialization or source-index writes
// may occur between the two actual address computations.
bool EquivalentFullBases(unsigned a,unsigned b,
                         const std::map<unsigned,Location>& locations,
                         const std::vector<uint32_t>& words) {
  if(a==b) return true;
  const unsigned low=std::min(a,b),high=std::max(a,b);
  if(!UnconditionalFetchOnlyInterval(low,high,locations,words)) return false;
  const unsigned p=a*3,q=b*3;
  constexpr uint32_t inherited0=0xC7F00FE0u;  // src/relative, constant selector, src swizzle
  if((words[p]&inherited0)!=(words[q]&inherited0) ||
     (words[p+1]&0x80008000u)!=(words[q+1]&0x80008000u) ||
     (words[p+2]&0x800000FFu)!=(words[q+2]&0x800000FFu)) return false;
  // Relative TEMP addressing needs a broader loop/aL proof; intentionally out
  // of scope. Likewise a relative fetch destination could overwrite the source.
  if(words[p]&0x800u) return false;
  const unsigned source=(words[p]>>5)&63, component=(words[p]>>30)&3;
  for(unsigned i=low;i<high;++i) {
    const unsigned r=i*3;
    if(words[r]&0x40000u) return false;
    if(((words[r]>>12)&63)==source &&
       (me::native::VertexFetchWrittenMask(words[r+1]&4095)&(1u<<component))) return false;
  }
  return true;
}
// Bounded exception to same-EXEC: two immediately adjacent unconditional EXECs,
// with consecutive instruction ranges. The entire moved interval is an
// unpredicated, unserialized fetch-only region inheriting proven equivalent bases.
bool AdjacentFetchOnlyRegion(unsigned old, unsigned next,
                             const std::map<unsigned,Location>& locations,
                             const std::vector<uint32_t>& words) {
  const auto& a=locations.at(old); const auto& b=locations.at(next);
  const auto& first=a.clause<b.clause?a:b;
  const auto& second=a.clause<b.clause?b:a;
  if(second.clause!=first.clause+1 || first.opcode!=1 || second.opcode!=1 ||
     first.clause_first+first.clause_count!=second.clause_first) return false;
  const unsigned low=std::min(old,next),high=std::max(old,next);
  if(!UnconditionalFetchOnlyInterval(low,high,locations,words)) return false;
  for(unsigned i=low;i<=high;++i) {
    const auto it=locations.find(i);
    if(it==locations.end() || !EquivalentFullBases(it->second.cluster,a.cluster,locations,words) ||
       (it->second.clause!=a.clause && it->second.clause!=b.clause) ||
       it->second.serialized || (words[i*3+1]&0x80000000u)) return false;
  }
  return true;
}
#ifndef O_BINARY
#define O_BINARY 0  // POSIX has no text mode
#endif
void ExclusiveWrite(const char* path, const Bytes& b) {
  // O_BINARY: on Windows open() defaults to text mode and turns every 0x0A byte into 0D 0A, corrupting the container.
  const int fd=open(path,O_WRONLY|O_CREAT|O_EXCL|O_BINARY,0600);
  if(fd<0) Reject("exclusive output creation failed: "+std::string(std::strerror(errno)));
  size_t written=0;
  while(written<b.size()) {
    const ssize_t n=write(fd,b.data()+written,b.size()-written);
    if(n<=0) {close(fd); Reject("output write failed (incomplete fresh output retained)");}
    written+=size_t(n);
  }
  if(close(fd)) Reject("output close failed");
}
}  // namespace

int main(int argc,char** argv) try {
  if(argc!=4) {std::fprintf(stderr,"usage: %s template.bin actual.bin fresh-output.bin\n",argv[0]);return 2;}
  Container original(argv[1]), actual(argv[2]);
  std::printf("sources template=%s actual=%s code-offset=%zu code-words=%zu DECL-count=%zu\n",
              argv[1],argv[2],actual.start,actual.words.size(),actual.entries.size());
  if(original.bytes.size()!=actual.bytes.size() || original.start!=actual.start ||
     original.size!=actual.size || original.decl!=actual.decl || original.entries!=actual.entries)
    Reject("container layout/DECL differs before repair");
  for(size_t i=0;i<original.bytes.size();++i)
    if((i<original.start || i>=original.start+original.size) && original.bytes[i]!=actual.bytes[i])
      Reject("non-code metadata differs before repair");
  std::set<unsigned> declared;
  for(uint32_t e:original.entries) {
    const unsigned i=e&4095;
    if(uint64_t(i)*3+3>original.words.size() || (original.words[i*3]&31)!=0)
      Reject("DECL is not an in-bounds vertex fetch");
    declared.insert(i);
  }
  const auto locations=FetchLocations(actual.words);
  if(locations.size()!=declared.size()) Reject("undeclared or missing executed vertex fetch");
  for(const auto& [i,l]:locations) if(!declared.contains(i)) Reject("fetch address set changed");
  for(size_t i=0;i<original.words.size();++i)
    if(!declared.contains(unsigned(i/3)) && original.words[i]!=actual.words[i])
      Reject("NON-fetch ALU/CF differs at word "+std::to_string(i));
  std::printf("proof all non-fetch ALU/CF words and all pre-repair metadata identical; fetch normalization masks=0007FFFF/80000FFF/80000000 with representable swizzle\n");
  constexpr uint32_t masks[3]{0x0007FFFFu,0x80000FFFu,0x80000000u};
  std::map<unsigned,unsigned> mapping; std::set<unsigned> used;
  for(unsigned old:declared) {
    const size_t p=old*3; const unsigned temp=(original.words[p]>>12)&63;
    const uint32_t sw=original.words[p+1]&4095;
    std::vector<unsigned> matches;
    for(unsigned candidate:declared) {
      const size_t q=candidate*3;
      if(((actual.words[q]>>12)&63)!=temp ||
         me::native::VertexFetchWrittenMask(actual.words[q+1]&4095)!=me::native::VertexFetchWrittenMask(sw) ||
         !me::native::VertexFetchSwizzleRepresentable(sw,actual.words[q+1]&4095)) continue;
      if((actual.words[q]&masks[0])!=(original.words[p]&masks[0]) ||
         (actual.words[q+1]&0x80000000u)!=(original.words[p+1]&0x80000000u) ||
         (actual.words[q+2]&masks[2])!=(original.words[p+2]&masks[2])) continue;
      matches.push_back(candidate);
    }
    if(matches.size()!=1 || !used.insert(matches[0]).second) Reject("fetch mapping is not unique bijection");
    mapping[old]=matches[0];
    std::printf("proof DECL instruction %u -> %u TEMP r%u write-mask=%X clause=%u->%u full-base=%u->%u\n",
      old,matches[0],temp,me::native::VertexFetchWrittenMask(sw),locations.at(old).clause,
      locations.at(matches[0]).clause,locations.at(old).cluster,locations.at(matches[0]).cluster);
  }
  for(const auto& [old,next]:mapping) {
    if(old!=next && !UnconditionalFetchOnlyInterval(std::min(old,next),std::max(old,next),locations,actual.words))
      Reject("DECL permutation crosses non-fetch/predicate/serialized interval");
    if(!EquivalentFullBases(locations.at(old).cluster,locations.at(next).cluster,locations,actual.words))
      Reject("DECL permutation crosses inequivalent actual full/mini bases: "+std::to_string(old)+"->"+std::to_string(next));
    if(locations.at(old).cluster!=locations.at(next).cluster)
      std::printf("proof equivalent full bases %u/%u: SAME const/index/stride/round, no index writes/ALU/predicate/barriers\n",
                  locations.at(old).cluster,locations.at(next).cluster);
    if(locations.at(old).clause!=locations.at(next).clause) {
      if(!AdjacentFetchOnlyRegion(old,next,locations,actual.words))
        Reject("DECL permutation crosses unproven EXEC boundary: "+std::to_string(old)+"->"+std::to_string(next));
      std::printf("proof adjacent unconditional EXEC fetch-only region %u..%u, equivalent actual full base %u\n",
                  std::min(old,next),std::max(old,next),locations.at(old).cluster);
    }
  }
  Bytes output=actual.bytes;
  for(size_t i=0;i<original.entries.size();++i) {
    const uint32_t e=original.entries[i]; Put(output,actual.decl+i*4,(e&~4095u)|mapping.at(e&4095));
  }
  if(!std::equal(output.begin()+actual.start,output.begin()+actual.start+actual.size,
                 actual.bytes.begin()+actual.start)) Reject("internal code preservation failed");
  ExclusiveWrite(argv[3],output);
  std::printf("ACCEPT: unique bijection, proven EXEC/base continuity, all ALU/CF and metadata unchanged; output=%s bytes=%zu\n",argv[3],output.size());
  return 0;
} catch(const std::exception& e) {
  std::fprintf(stderr,"REJECT: %s; no accepted output produced\n",e.what()); return 1;
}
