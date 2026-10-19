// Standalone production-policy regression: clang++ -std=c++20 -O2 this.cpp.
// Includes the exact offline proof implementation; no Vulkan or private assets.
#define main RepairVertexVariantMain
#include "../tools/repair_vertex_variant_declarations.cpp"
#undef main
#include <filesystem>

void Check(bool value,const char* why) { if(!value) throw std::runtime_error(why); }
struct Fixture {
  Bytes original=Bytes(248),actual;
  Fixture(bool second_full) {
    Put(original,0,0x102A1101); Put(original,4,128); Put(original,8,120); Put(original,24,32);
    Put(original,32,0); Put(original,36,120); Put(original,60,7);
    const uint64_t cf0=uint64_t(1)<<44 | uint64_t(0x555)<<16 | uint64_t(6)<<12 | 2;
    const uint64_t cf1=uint64_t(1)<<44 | uint64_t(1)<<16 | uint64_t(2)<<12 | 8;
    Put(original,128,uint32_t(cf0)); Put(original,132,uint32_t(cf0>>32)|uint32_t(cf1<<16));
    Put(original,136,uint32_t(cf1>>16));
    const unsigned temps[7]{7,6,2,3,4,5,1};
    const uint32_t sw[7]{0xE0A,0xE0A,0xE42,0xE11,0x688,0x688,0x23F};
    for(unsigned i=0;i<7;++i) {
      Put(original,68+i*4,((i+1)<<12)|(i+2));
      Put(original,128+(i+2)*12,0x05F80000|(temps[i]<<12));
      Put(original,128+(i+2)*12+4,sw[i]);
    }
    Put(original,128+9*12,0x12345678); Put(original,128+9*12+4,0x11223344);
    actual=original;
    const unsigned from[7]{0,1,2,3,6,4,5};
    for(unsigned i=0;i<7;++i) {
      Put(actual,128+(i+2)*12,0x05F80000|(temps[from[i]]<<12));
      Put(actual,128+(i+2)*12+4,sw[from[i]] | (i ? 0x40000000u : 0));
      Put(actual,128+(i+2)*12+8,(i<<8)|10);
    }
    if(second_full) Put(actual,128+7*12+4,sw[4]);
  }
};
int main() try {
  char pattern[]="/tmp/me-decl-regression-XXXXXX";
  const char* directory=mkdtemp(pattern); Check(directory,"mkdtemp");
  const std::string dir=directory;
  auto invoke=[&](Bytes original,Bytes actual,const std::string& name,int expected) {
    const std::string o=dir+"/"+name+"-original.bin",a=dir+"/"+name+"-actual.bin",r=dir+"/"+name+"-repaired.bin";
    ExclusiveWrite(o.c_str(),original); ExclusiveWrite(a.c_str(),actual);
    std::string tool="repair"; char* args[]{tool.data(),const_cast<char*>(o.c_str()),const_cast<char*>(a.c_str()),const_cast<char*>(r.c_str())};
    Check(RepairVertexVariantMain(4,args)==expected,name.c_str());
    if(expected) Check(!std::filesystem::exists(r),"rejection produced output");
    else {
      const Bytes out=Read(r.c_str());
      Check(out.size()==actual.size(),"output extent");
      for(size_t i=0;i<out.size();++i)
        if(i!=87 && i!=91 && i!=95) Check(out[i]==actual[i],"code/ALU/CF/header/DECL highbits preservation");
      Check(Be(out,84)==((Be(original,84)&~4095u)|7u),"indices mapping");
      Check(Be(out,88)==((Be(original,88)&~4095u)|8u),"weights mapping");
      Check(Be(out,92)==((Be(original,92)&~4095u)|6u),"UV mapping");
      Check(RepairVertexVariantMain(4,args)==1,"exclusive output must reject overwrite");
    }
  };
  Fixture f(true); invoke(f.original,f.actual,"equivalent-bases",0);
  Fixture one(false); invoke(one.original,one.actual,"one-base-adjacent-exec",0);
  const std::string temp=dir+"/probe.bin"; ExclusiveWrite(temp.c_str(),f.actual);
  Container c(temp.c_str()); const auto locations=FetchLocations(c.words);
  Check(EquivalentFullBases(2,7,locations,c.words),"baseline base equivalence");
  const std::array<std::pair<unsigned,uint32_t>,8> mutations{{
    {7*3,1u<<20}, {7*3,1u<<5}, {7*3+2,1u}, {7*3,1u<<11},
    {7*3+1,1u<<15}, {7*3+1,1u<<31}, {7*3,1u<<30}, {7*3+2,1u<<31}}};
  for(const auto& [word,bit]:mutations) {
    auto w=c.words; w[word]^=bit;
    Check(!EquivalentFullBases(2,7,locations,w),"inherited const/index/stride/relative/round/predicate mutant");
  }
  {auto w=c.words; w[3*3]&=~(63u<<12); Check(!EquivalentFullBases(2,7,locations,w),"intermediate source-index write");}
  {auto l=locations; l[4].opcode=3; Check(!EquivalentFullBases(2,7,l,c.words),"conditional interval");}
  {auto l=locations; l[4].serialized=true; Check(!EquivalentFullBases(2,7,l,c.words),"serialized interval");}
  {auto l=locations; l.erase(4); Check(!EquivalentFullBases(2,7,l,c.words),"ALU/gap interval");}
  {auto w=c.words; w[4*3]|=1u<<18; Check(!EquivalentFullBases(2,7,locations,w),"relative destination uncertainty");}
  {auto a=f.actual; Put(a,128+9*12,Be(a,128+9*12)^1); invoke(f.original,a,"stale-ALU",1);}
  {auto a=f.actual; Put(a,44,Be(a,44)^1); invoke(f.original,a,"stale-header",1);}
  {auto a=f.actual; Put(a,84,Be(a,84)^0x1000); invoke(f.original,a,"changed-DECL-ABI",1);}
  {auto a=f.actual; Put(a,128+7*12,Be(a,128+7*12)^(1u<<20)); invoke(f.original,a,"different-buffer-selector",1);}
  std::printf("PASS: production DECL repair, all inherited-base mutants, barriers/index writes, metadata/code preservation, exclusive fresh output. Fixtures retained %s\n",directory);
  return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
