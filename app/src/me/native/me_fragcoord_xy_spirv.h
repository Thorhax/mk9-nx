#pragma once
#include <cstdint>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>

namespace me::native {
// This is the inverse of the current native raster coordinate modification,
// not Xenos paramgen conformance: the native renderer consumes continuous iPos.xy-0.5,
// whereas the SDK paramgen path floors FragCoord. Do not add floor here.
// Enum value MUST participate in the final module cache key, along with full
// source code. Compose ONCE, on the selected material, BEFORE depth transforms:
// those transforms inject raw FragCoord loads that must remain unmodified.
enum class GuestFragCoordXYMode : uint32_t { GridX2 = 1, Phase1 = 2, Phase2 = 3 };

inline bool TransformGuestFragCoordXY(const std::vector<uint32_t>& input,
                                     std::vector<uint32_t>& output,
                                     std::string& reason, GuestFragCoordXYMode mode) {
  if (&input == &output) { reason = "input/output alias"; return false; }
  output.clear(); reason.clear();
  const auto fail = [&](const char* why) { output.clear(); reason = why; return false; };
  if (mode != GuestFragCoordXYMode::GridX2 && mode != GuestFragCoordXYMode::Phase1 &&
      mode != GuestFragCoordXYMode::Phase2) return fail("unsupported XY mode");
  if (input.size() < 5 || input[0] != 0x07230203 || !input[3] || input[4])
    return fail("invalid SPIR-V header");
  struct Ins { size_t at; uint32_t count, op; };
  struct Ptr { uint32_t storage, type; };
  struct Var { uint32_t type, storage, count; };
  std::vector<Ins> instructions;
  std::map<uint32_t,uint32_t> floats, ints, constants;
  std::map<uint32_t,std::pair<uint32_t,uint32_t>> vectors;
  std::map<uint32_t,Ptr> types;
  std::map<uint32_t,Var> vars;
  uint32_t coord = 0, entry = 0, scalar = 0, vec4 = 0;
  size_t first_function = input.size(); bool in_function = false;
  for (size_t at = 5; at < input.size();) {
    const auto* w = input.data()+at; uint32_t n=w[0]>>16, op=w[0]&65535;
    if (!n || n>input.size()-at) return fail("malformed instruction");
    instructions.push_back({at,n,op});
    if (op==15) { if(n<4 || entry || w[1]!=4) return fail("requires one fragment entry"); entry=w[2]; }
    else if(op==22 && n==3) floats[w[1]]=w[2];
    else if(op==21 && n==4) ints[w[1]]=w[2];
    else if(op==23 && n==4) vectors[w[1]]={w[2],w[3]};
    else if(op==32 && n==4) types[w[1]]={w[2],w[3]};
    else if(op==43 && n==4 && ints.count(w[1]) && ints[w[1]]==32) constants[w[2]]=w[3];
    else if(op==71 && n>=4 && w[2]==11 && w[3]==15) {
      if(n!=4 || coord) return fail("duplicate or malformed FragCoord"); coord=w[1];
    } else if(op==72 && n>=5 && w[3]==11 && w[4]==15) return fail("FragCoord block unsupported");
    else if(op==74 || op==75) return fail("group decorations unsupported");
    if(op==54) { in_function=true; if(first_function==input.size()) first_function=at; }
    if(op==59 && !in_function) { if(n<4) return fail("malformed variable"); vars[w[2]]={w[1],w[3],n}; }
    at+=n;
  }
  if(!entry || first_function==input.size()) return fail("missing entry/functions");
  if(!coord) { output=input; return true; }
  if(!vars.count(coord) || !types.count(vars[coord].type)) return fail("FragCoord is not direct variable");
  const auto p=types[vars[coord].type];
  if(vars[coord].storage!=1 || vars[coord].count!=4 || p.storage!=1 || !vectors.count(p.type) ||
     vectors[p.type].second!=4 || floats[vectors[p.type].first]!=32)
    return fail("FragCoord must be uninitialized Input vec4<float32>");
  vec4=p.type; scalar=vectors[vec4].first;
  // Kind -1 means whole vector; 0..3 are constant components.
  std::map<uint32_t,int> pointers{{coord,-1}};
  for(const auto& ins:instructions) {
    if(ins.at<first_function || (ins.op!=65 && ins.op!=66)) continue;
    const auto* w=input.data()+ins.at;
    if(ins.count<4) return fail("malformed access chain");
    if(!pointers.count(w[3])) continue;
    if(ins.count!=5 || pointers[w[3]]!=-1 || !constants.count(w[4]) || constants[w[4]]>3)
      return fail("dynamic/nested FragCoord pointer unsupported");
    if(!types.count(w[1]) || types[w[1]].storage!=1 || types[w[1]].type!=scalar)
      return fail("invalid component pointer");
    pointers[w[2]]=int(constants[w[4]]);
  }
  for(const auto& ins:instructions) {
    if(ins.at<first_function) continue;
    const auto* w=input.data()+ins.at;
    if(ins.op==8 || ins.op==317) continue; // debug file/line literals
    for(uint32_t k=1;k<ins.count;++k) {
      // Literal positions cannot be treated as IDs (FragCoord commonly ID3).
      if((ins.op==79 && k>=5)||(ins.op==81 && k>=4)||(ins.op==82 && k>=5)||
         (ins.op==12 && k==4)||(ins.op==54 && k==3)||(ins.op==246 && k>=3)||
         (ins.op==247 && k==2)||(ins.op==250 && k>=4)||
         (ins.op==251 && k>=3 && (k&1))||(ins.op==61 && k>=4)||
         (ins.op==59 && k==3)) continue; // StorageClass is a literal, not an ID.
      if(!pointers.count(w[k])) continue;
      const bool load=ins.op==61 && k==3 && ins.count>=4 &&
          w[1]==(pointers[w[k]]==-1?vec4:scalar);
      const bool chain=(ins.op==65 || ins.op==66) && (k==2 || k==3) && pointers.count(w[2]);
      if(!load && !chain) return fail("FragCoord pointer escape/unsupported use");
    }
  }
  if(uint64_t(input[3])+instructions.size()*8ull+3>UINT32_MAX) return fail("ID bound overflow");
  uint32_t next=input[3], cx=next++,cy=next++;
  const auto emit=[](std::vector<uint32_t>& dst,uint32_t op,std::initializer_list<uint32_t> a) {
    dst.push_back((uint32_t(a.size()+1)<<16)|op); dst.insert(dst.end(),a);
  };
  const bool grid=mode==GuestFragCoordXYMode::GridX2;
  std::vector<uint32_t> arithmetic;
  size_t annotation_at=0;
  output.assign(input.begin(),input.begin()+5);
  for(const auto& ins:instructions) {
    const auto* w=input.data()+ins.at;
    if(!annotation_at && ins.op>=19 && ins.op<=39) annotation_at=output.size();
    if(ins.at==first_function) {
      emit(output,43,{scalar,cx,grid?0x3F000000u:0x3E800000u});
      emit(output,43,{scalar,cy,mode==GuestFragCoordXYMode::Phase2?0xBE800000u:0x3E800000u});
    }
    if(ins.op==61 && ins.count>=4 && pointers.count(w[3]) &&
       (pointers[w[3]]==-1 || pointers[w[3]]==0 || (!grid && pointers[w[3]]==1))) {
      const int kind=pointers[w[3]]; const uint32_t raw=next++;
      size_t at=output.size();output.insert(output.end(),w,w+ins.count);output[at+2]=raw;
      if(kind==-1) {
        const uint32_t x=next++,gx=next++,vx=grid?w[2]:next++;
        emit(output,81,{scalar,x,raw,0});emit(output,grid?133:131,{scalar,gx,x,cx});arithmetic.push_back(gx);
        emit(output,82,{vec4,vx,gx,raw,0});
        if(!grid) { const uint32_t y=next++,gy=next++;
          emit(output,81,{scalar,y,raw,1});emit(output,131,{scalar,gy,y,cy});arithmetic.push_back(gy);
          emit(output,82,{vec4,w[2],gy,vx,1}); }
      } else { emit(output,grid?133:131,{scalar,w[2],raw,kind==0?cx:cy});arithmetic.push_back(w[2]); }
    } else output.insert(output.end(),w,w+ins.count);
  }
  if(!annotation_at) return fail("missing type section");
  std::vector<uint32_t> decorations;
  for(uint32_t id:arithmetic) emit(decorations,71,{id,42}); // NoContraction
  output.insert(output.begin()+annotation_at,decorations.begin(),decorations.end());
  output[3]=next;return true;
}
} // namespace me::native
