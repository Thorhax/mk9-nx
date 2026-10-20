// banjo-nx: UnleashedRecomp-NX's plume Switch patches call LSFG-VK frame generation through plume_lsfg_bridge.h. banjo-nx
// does not ship LSFG-VK: these stand-ins decline (create returns no runtime), so ConfigureSwitchFrameGeneration is never
// enabled and every frame presents normally.
#include "plume_lsfg_bridge.h"

extern "C" {
LsfgNxRuntime *lsfg_nx_create(const LsfgNxCreateInfo *) { return nullptr; }
void lsfg_nx_destroy(LsfgNxRuntime *) {}
bool lsfg_nx_present(LsfgNxRuntime *, VkQueue, const VkPresentInfoKHR *, VkResult *) { return false; }
}
