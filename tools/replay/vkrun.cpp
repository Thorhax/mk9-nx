// mk9-nx (2026-10-23): runs one compute shader once on the PC through Vulkan, for tools/posdiag_replay.py.
//   vkrun <shader.spv> <binding>:<ubo|ssbo>:<file> ... -o <binding>:<out file>
// Every binding is in set 0; the -o binding is read back after the dispatch (1 x 1 x 1) into <out file>.
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { std::fprintf(stderr, "%s: %d\n", #x, r_); std::exit(1); } } while (0)

static std::vector<char> ReadFile(const char *path) {
  FILE *f = std::fopen(path, "rb");
  if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
  std::fseek(f, 0, SEEK_END);
  std::vector<char> d(std::ftell(f));
  std::fseek(f, 0, SEEK_SET);
  if (!d.empty() && std::fread(d.data(), 1, d.size(), f) != d.size()) std::exit(1);
  std::fclose(f);
  return d;
}

struct Binding { uint32_t index; bool ubo; std::vector<char> data; VkBuffer buffer; VkDeviceMemory memory; };

int main(int argc, char **argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: vkrun shader.spv binding:ubo|ssbo:file ... -o binding:out\n"); return 2; }
  std::vector<char> spv = ReadFile(argv[1]);
  std::vector<Binding> bindings;
  uint32_t out_binding = ~0u; std::string out_path;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-o" && i + 1 < argc) {
      std::string o = argv[++i];
      out_binding = std::stoul(o.substr(0, o.find(':')));
      out_path = o.substr(o.find(':') + 1);
      continue;
    }
    size_t c1 = a.find(':'), c2 = a.find(':', c1 + 1);
    Binding b{};
    b.index = std::stoul(a.substr(0, c1));
    b.ubo = a.substr(c1 + 1, c2 - c1 - 1) == "ubo";
    b.data = ReadFile(a.substr(c2 + 1).c_str());
    if (b.data.size() < 16) b.data.resize(16);
    bindings.push_back(std::move(b));
  }

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  VkInstance inst; CHECK(vkCreateInstance(&ici, nullptr, &inst));
  uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, nullptr);
  if (!n) { std::fprintf(stderr, "no Vulkan device\n"); return 1; }
  std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst, &n, pds.data());
  VkPhysicalDevice pd = pds[0];
  VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props);
  std::fprintf(stderr, "vkrun: %s\n", props.deviceName);
  uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
  std::vector<VkQueueFamilyProperties> qf(qn); vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
  uint32_t qfi = 0; while (qfi < qn && !(qf[qfi].queueFlags & VK_QUEUE_COMPUTE_BIT)) ++qfi;
  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = qfi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
  VkDevice dev; CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
  VkQueue q; vkGetDeviceQueue(dev, qfi, 0, &q);
  VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);

  std::vector<VkDescriptorSetLayoutBinding> lb;
  for (Binding &b : bindings) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = b.data.size();
    bci.usage = b.ubo ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    CHECK(vkCreateBuffer(dev, &bci, nullptr, &b.buffer));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, b.buffer, &mr);
    uint32_t type = 0;
    for (; type < mp.memoryTypeCount; ++type)
      if ((mr.memoryTypeBits & (1u << type)) && (mp.memoryTypes[type].propertyFlags &
          (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
          (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) break;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size; mai.memoryTypeIndex = type;
    CHECK(vkAllocateMemory(dev, &mai, nullptr, &b.memory));
    CHECK(vkBindBufferMemory(dev, b.buffer, b.memory, 0));
    void *p; CHECK(vkMapMemory(dev, b.memory, 0, VK_WHOLE_SIZE, 0, &p));
    std::memcpy(p, b.data.data(), b.data.size()); vkUnmapMemory(dev, b.memory);
    VkDescriptorSetLayoutBinding l{};
    l.binding = b.index; l.descriptorCount = 1; l.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    l.descriptorType = b.ubo ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    lb.push_back(l);
  }
  VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dslci.bindingCount = uint32_t(lb.size()); dslci.pBindings = lb.data();
  VkDescriptorSetLayout dsl; CHECK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
  VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));
  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = spv.size(); smci.pCode = reinterpret_cast<const uint32_t *>(spv.data());
  VkShaderModule sm; CHECK(vkCreateShaderModule(dev, &smci, nullptr, &sm));
  VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpci.stage.module = sm; cpci.stage.pName = "main";
  cpci.layout = pl;
  VkPipeline pipe; CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));
  VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16}};
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1; dpci.poolSizeCount = 2; dpci.pPoolSizes = ps;
  VkDescriptorPool dp; CHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dp));
  VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = dp; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
  VkDescriptorSet ds; CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));
  std::vector<VkDescriptorBufferInfo> infos(bindings.size());
  std::vector<VkWriteDescriptorSet> writes(bindings.size());
  for (size_t i = 0; i < bindings.size(); ++i) {
    infos[i] = {bindings[i].buffer, 0, VK_WHOLE_SIZE};
    writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[i].dstSet = ds; writes[i].dstBinding = bindings[i].index; writes[i].descriptorCount = 1;
    writes[i].descriptorType = bindings[i].ubo ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  vkUpdateDescriptorSets(dev, uint32_t(writes.size()), writes.data(), 0, nullptr);
  VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpi.queueFamilyIndex = qfi;
  VkCommandPool cp; CHECK(vkCreateCommandPool(dev, &cpi, nullptr, &cp));
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = cp; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
  VkCommandBuffer cb; CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));
  VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  CHECK(vkBeginCommandBuffer(cb, &cbbi));
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
  vkCmdDispatch(cb, 1, 1, 1);
  CHECK(vkEndCommandBuffer(cb));
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1; si.pCommandBuffers = &cb;
  CHECK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
  CHECK(vkQueueWaitIdle(q));
  for (Binding &b : bindings) {
    if (b.index != out_binding) continue;
    void *p; CHECK(vkMapMemory(dev, b.memory, 0, VK_WHOLE_SIZE, 0, &p));
    FILE *f = std::fopen(out_path.c_str(), "wb");
    std::fwrite(p, 1, b.data.size(), f); std::fclose(f);
  }
  return 0;
}
