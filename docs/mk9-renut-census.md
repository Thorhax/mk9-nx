# MK9's Direct3D for Renut's renderer (the deko3d route)

Started 2026-10-20, following sa2-nx's `docs/deko3d-renderer-guide.md` (section 2, the census and the device layout).
MK9 links Direct3D 2.0.20353 (SA2 2.0.21076, N&B 2.0.7645). Neither SA2's nor N&B's image is on this machine, so the
functions were found by what they do (kernel imports, device fields, PM4 headers, call shapes), each named with
the evidence. Listings: `out/d3d_listing.txt` (every function in 0x82BD0000-0x82C20000, from the generated code).

## The hooks (MK9, SA2, N&B)

| Function | MK9 | SA2 | N&B | Evidence (MK9) |
|---|---|---|---|---|
| Direct3D_CreateDevice | 82BF6108 | 8254FC10 | 8264DF28 | UE3's sub_826B98A8 calls it; allocates 24448 (0x5F80) aligned 128; type 2 sets 11068 bit 0x80 |
| hardware path (InitializeHardwareDevice) | 82C019D0 | 8255CDA0 | 82659F30 | critical sections, VdGlobalDevice, fence words 11024/11028, SetRingBufferParameters |
| device block init | 82C01960 | 8255C808 | | +60 = 1, qword 0x8339BF90 -> 11008, thread 11016, title 11020, side buffer 4800 -> 17008 |
| XenonGpu_Initialize | 82BFB8D0 | | 82663208 | type-2 path, then InitializeApiState |
| type-2 tail | 82C01C38 | | | 21840/21844/15212 = -1, sub_82C09370, InitializeApiState |
| InitializeApiState | 82C01260 | 8255C150 | 826597C8 | |
| InitializePresentationParameters | 82C014C8 | 8255CB20 | 82659AA8 | (CDevice*, D3DPRESENT_PARAMETERS*), result tested |
| CDevice::KickOff | 82BFCD60 | 8254EBB8 | 82210AB0 | returns m_pRing (+48); in-KickOff word 13472; SetFence path (11069 bit 2, BlockOnFence(11036 - 2)) |
| CDevice::KickOffSegment | 82BFCBD8 | 8254E830 | 822106C8 | writes the segment's INDIRECT_BUFFER (0x8100xxxx type-0 pair / +13632 list); ends in the segment allocator |
| ring segment allocator | 82BFC0B0 | 8254DA90 | - | KickOffSegment, BeginRingBig, Swap call it |
| BeginRingBig | 82BFCFA0 | 8254EDF8 | 821F7C18 | (dev, dwords): KickOff, then the allocator, returns m_pRing |
| RingBufferDeviceAllocate | ? | 8254D4B8 | 8222D3E0 | to find |
| AllocateInlineData | 82BFBB78? | 8254D9A8 | 8222D240 | (dev, dwords, alignment): BeginVertices and BeginIndexedVertices call it; to confirm |
| SetRingBufferParameters | 82BFD0E8 | 8254EF40 | 82657508 | only caller of VdInitializeRingBuffer; ring buffers at 15112/15116 |
| CDevice::BlockOnFence | 82BFBC60 | 8254D6E8 | 8228F558 | (dev, fence, type, -, nokick in r7): 11024/11036, 13472, KickOff, CBlocker loop |
| CBlocker::Check | 82BF5680 | 8254F400 | 8264DCD0 | the GPU wait iteration (mk9_gpu_wait.cpp hooks it today) |
| IsFencePending | 82BFCEA0 | 8254ECF8 | 8228F650 | (fence): device from VdGlobalDevice (0x82000658), KickOff if current |
| InsertCallback | 82BFD090 | 8254EEE8 | 821FB688 | (dev, type, callback, context), just before SetRingBufferParameters as in SA2 |
| InsertFence | 82BFCE68 | | | 11036 -> 11056, KickOff |
| SynchronizeToPresentationInterval | 82BF6928 | 825589D8 | 8228A748 | the only writer of 11069 bit 0x10 (Swap clears it) |
| Swap | 82BF6D58 | 82558E98 | 82287590 | only VdSwap caller |
| DrawVertices | 82BF09E0 | 82565C60 | 8222C7A0 | (dev, prim, start, count), dirty-mask flush first |
| DrawIndexedVertices | 82BF0DF8 | 82566078 | 826567C8 | (dev, prim, base, start, count), 8 callers |
| DrawVerticesUP | 82BF03B8 | 82565C18 | 8222E520 | BeginVertices + memcpy + restore m_pRing from 13716 |
| BeginVertices | 82BEFEF8 | 82565758 | 8222CC50 | stores the inline end at 13716, stride / 4 at 12776 |
| EndVertices (inline) | 82BEFEE8 | | | m_pRing = [13716] |
| DrawIndexedVerticesUP | 82BF0940 | | 82656728 | BeginIndexedVertices + two memcpys |
| BeginIndexedVertices | 82BF0400 | | | |
| Draw(Indexed)TessellatedVertices? | 82BF1270, 82BF1710 | | 82230838 | write VGT_HOS fields (10612); 82BF1270 has no direct caller |
| ClearF | 82BEF680 | 82563978 | 822811A0 | (dev, flags, rect, color, f1 Z, r8 stencil); reads the depth surface 12704 |
| Clear | 82BEFD00 | | 822812C8 | ClearF's only caller |
| Resolve | 82BFB4C8 | 825656F8 | 8222D6C8 | public wrapper of sub_82BFA378; UE3 and Present call it; stack args at 252/260 |
| XGRegisterVertexShader | 82BEB740 | 8255E658 | 8264E7B8 | (shader, physical): +0x20 = physical; the container at +0x368 (sub_82BEB700 copies it) |
| XGRegisterPixelShader | 82BE9B98 | 8255DBF8 | 8264E670 | (shader, physical): +0x18 = physical; container at +0x28 (sub_82BEB5D8) |
| CreateVertexShader / CreatePixelShader | inlined | 8255E750 / 8255E568 | | UE3's sub_826C07C0 / sub_826C0978 |
| CreateVertexDeclaration | 82BEA068 | 8255E198 | 8264EA90 | (elements: 12-byte D3DVERTEXELEMENT9 up to Stream 0xFF) |
| D3DResource_Release | 82BE58A8 | | | atomic -1 at +4, then Destroy |
| D3DResource_Destroy | 82BE5318 | 8254A200 | 822084F0 | |
| SetDepthStencilSurface | 82BF52A8 | | | stores 12704; targets 12688-12700; tiling surfaces 12976-12992 |

## The device layout

MK9's private layout is SA2's with the bound objects and everything after them 128 bytes earlier. The block
11008-11072 (fences, flags) is at SA2's offsets. The device is 0x5F80 bytes (SA2 0x6080, N&B 0x5E00).

| Field | MK9 | SA2 | Read in (MK9) |
|---|---|---|---|
| global qword copy | 11008 (from 0x8339BF90) | 11008 (0x82980268) | sub_82C01960 |
| creation thread / title process | 11016 / 11020 | same | sub_82C01960 |
| completed-fence word pointer, swap scratch | 11024 / 11028 | same | hardware path (96 and 32 bytes) |
| fence tag / swap fence tag | 11036 / 11056 | same | BlockOnFence, InsertFence |
| flags | 11068-11071 | same | CreateDevice, KickOff, Swap |
| vertex declaration | 12112? | 12240 | to confirm |
| index buffer | 12684 | 12812 | sub_82BF1710 |
| render targets [4], depth | 12688, 12704 | 12816, 12832 | SetDepthStencilSurface sub_82BF52A8, ClearF |
| stream sources, strides | 12708?, 12776 | 12836, 12904 | BeginVertices (strides) |
| textures | 12792? | 12920 | to confirm |
| pixel / vertex shader | 12940 / 12944 | 13068 / 13072 | sub_82BFEEE8 |
| tiling surfaces | 12976 | 13104 | SetDepthStencilSurface |
| in KickOff | 13472 | 13600 | KickOff, BlockOnFence |
| inline vertices' ring end | 13716 | 13844 | BeginVertices, EndVertices |
| command buffer being recorded | 13792 | 13920 | KickOffSegment |
| ring buffers (SetRingBufferParameters) | 15112 / 15116 | | sub_82BFD0E8 |
| -1 words | 21840, 21844, 15212 | 21968, 21972, 15340 | type-2 tail |
| critical sections | 15220, 15248 | 15348, 15376 | hardware path |
| present fence tag | 15216 | 15344 | Swap |
| swap count | 16840 | 16968 | Swap |
| side buffer | 17008 | 17136 | sub_82C01960 |
| timer frequency, reciprocal | 21868, 21872 | 21996, 22000 | hardware path |
| behaviour flags | 24264 | 24392 | CreateDevice |
| VdGlobalDevice import slot | 0x82000658 | 0x82000710 | the image's import record 0x000101BE |

## Open

- RingBufferDeviceAllocate, AllocateInlineData (confirm), the window scissor writer (SA2's sub_8254FCE8), the gamma
  ramp's field (SA2 +15408) and SetGammaRamp, the presentation interval copy (SA2 +14000).
- The native renderer's counters say only 7,590 of 51,118 draws in 10 s were paired with a Draw* record
  (mk9_001.log). Every DRAW_INDX header in the code is written inside Direct3D (0x82BE4AD0-0x82C089C0), so the
  others should be internal draws (Clear, Resolve, Present) or a counting artefact; to check on the PC once Renut
  draws.
- 18,880 shader containers (UE3), against SA2's 103: the shader table and the deko3d chain have to scale.
