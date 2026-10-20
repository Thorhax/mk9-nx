/*
 * sa2-nx (2026-10-20): the Vulkan entry points of a Switch build without Mesa's
 * NVK driver (REXGLUE_SWITCH_NO_NVK, rexglue_switch.cmake).
 *
 * Only four driver symbols are linked by name: vkGetInstanceProcAddr and
 * vkDestroyInstance (ui/vulkan/vulkan_instance.cpp) and the loader-interface
 * pair plume's Vulkan back end starts from (vk_icdNegotiateLoaderICDInterfaceVersion,
 * vk_icdGetInstanceProcAddr). Every other Vulkan call goes through the function
 * pointers those return. Here they find nothing, so creating a Vulkan instance
 * fails cleanly, and libvulkan.a (the whole driver) stays out of the NRO. Such a
 * build draws with a renderer that does not use Vulkan (the title's own, e.g.
 * deko3d); the SDK's Vulkan presenter and GPU plugin are left unusable.
 */
#include <errno.h>
#include <malloc.h>
#include <stddef.h>
#include <stdint.h>

/* The static driver also brought posix_memalign, which newlib declares and does
 * not define; FFmpeg's av_malloc needs it. memalign's blocks are freed with free. */
int posix_memalign(void **out, size_t alignment, size_t size) {
  if (alignment < sizeof(void *) || (alignment & (alignment - 1)) != 0) return EINVAL;
  void *p = memalign(alignment, size);
  if (!p && size) return ENOMEM;
  *out = p;
  return 0;
}

typedef void (*rex_vk_void_function)(void);

rex_vk_void_function vkGetInstanceProcAddr(void *instance, const char *name) {
  (void)instance;
  (void)name;
  return NULL;
}

void vkDestroyInstance(void *instance, const void *allocator) {
  (void)instance;
  (void)allocator;
}

rex_vk_void_function vk_icdGetInstanceProcAddr(void *instance, const char *name) {
  (void)instance;
  (void)name;
  return NULL;
}

/* VK_ERROR_INCOMPATIBLE_DRIVER: no driver to negotiate with. */
int32_t vk_icdNegotiateLoaderICDInterfaceVersion(uint32_t *version) {
  (void)version;
  return -9;
}
