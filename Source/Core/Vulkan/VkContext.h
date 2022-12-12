#pragma once

#include <set>
#include <unordered_map>

#include "Core/Memory.h"

#include "Graphics/Interface/IWindow.h"

#include <vulkan/vulkan.h>

#ifdef APIENTRY
#    undef APIENTRY
#endif

// clang-format off
#include <windows.h>
#include <vulkan/vulkan_win32.h>
// clang-format on

using namespace SE::Core;

namespace SE::Graphics::Internal
{
    enum class eBufferBindType : uint32_t
    {
        VERTEX_BUFFER  = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        INDEX_BUFFER   = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        STORAGE_BUFFER = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        UNIFORM_BUFFER = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        UNKNOWN        = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
    };

    class VkContext
    {
      public:
        VkContext() = default;
        ~VkContext();

        VkContext( Ref<IWindow> aWindow, bool aEnableValidation );

        VkBuffer CreateBuffer( VkBufferUsageFlags aBufferFlags, size_t aSize, bool aIsHostVisible, bool aIsCudaShareable );
        void     DestroyBuffer( VkBuffer aBuffer );

        VkImage CreateImage( uint32_t aWidth, uint32_t aHeight, uint32_t aDepth, uint32_t aMipLevels, uint32_t aLayers,
                             uint8_t aSampleCount, bool aIsCudaShareable, bool aCubeCompatible, VkFormat aFormat, VkMemoryPropertyFlags aProperties,
                             VkImageUsageFlags aUsage );
        void    DestroyImage( VkImage aBuffer );

        VkSampler CreateSampler( VkFilter aMinificationFilter, VkFilter aMagnificationFilter, VkSamplerAddressMode aWrappingMode,
                                 VkSamplerMipmapMode aMipmapMode );
        void      DestroySampler( VkSampler aImage );

        VkImageView CreateImageView( VkImage aImageObject, uint32_t aLayerCount, VkImageViewType aViewType, VkFormat aImageFormat,
                                     VkImageAspectFlags aAspectMask, VkComponentMapping aComponentSwizzle );
        void        DestroyImageView( VkImageView aImage );

        VkRenderPass CreateRenderPass( std::vector<VkAttachmentDescription> aAttachments, std::vector<VkSubpassDescription> aSubpasses,
                                       std::vector<VkSubpassDependency> aSubpassDependencies );
        void         DestroyRenderPass( VkRenderPass aRenderPass );

        VkFramebuffer CreateFramebuffer( std::vector<VkImageView> aImageViews, uint32_t aWidth, uint32_t aHeight, uint32_t aLayers,
                                         VkRenderPass aRenderPass );
        void          DestroyFramebuffer( VkFramebuffer aFramebuffer );

        VkPipelineLayout CreatePipelineLayout( std::vector<VkDescriptorSetLayout> aDescriptorSetLayout,
                                               std::vector<VkPushConstantRange>   aPushConstants );
        void             DestroyPipelineLayout( VkPipelineLayout aPipelineLayout );

        VkPipeline CreatePipeline( VkGraphicsPipelineCreateInfo aCreateInfo );
        void       DestroyPipeline( VkPipeline aPipeline );

        VkShaderModule CreateShaderModule( std::vector<uint32_t> aByteCode );
        void           DestroyShaderModule( VkShaderModule aShaderModule );

        VkDescriptorPool CreateDescriptorPool( uint32_t aDescriptorSetCount, std::vector<VkDescriptorPoolSize> aPoolSizes );
        void             DestroyDescriptorPool( VkDescriptorPool aDescriptorPool );
        VkDescriptorSet  AllocateDescriptorSet( VkDescriptorPool aDescriptorPool, VkDescriptorSetLayout aLayout,
                                                uint32_t aDescriptorCount = 0 );
        void FreeDescriptorSet( VkDescriptorPool aDescriptorPool, VkDescriptorSet *aDescriptorSet, uint32_t aDescriptorCount = 0 );

        VkDescriptorSetLayout CreateDescriptorSetLayout( std::vector<VkDescriptorSetLayoutBinding> aBindings, bool aUnbounded );
        void                  DestroyDescriptorSetLayout( VkDescriptorSetLayout aDescriptorSetLayout );

        std::tuple<VkFormat, uint32_t, VkExtent2D, VkSwapchainKHR> CreateSwapChain();
        void                                                       DestroySwapChain( VkSwapchainKHR aSwapchain );
        std::vector<VkImage>                                       GetSwapChainImages( VkSwapchainKHR aSwapChain );

        VkFence CreateFence();
        void    DestroyFence( VkFence aFence );
        void    ResetFences( std::vector<VkFence> aFences );
        void    ResetFence( VkFence aFence );
        void    WaitForFences( std::vector<VkFence> aFences, uint64_t aTimeout );
        void    WaitForFences( std::vector<VkFence> aFences );
        void    WaitForFence( VkFence aFence, uint64_t aTimeout );
        void    WaitForFence( VkFence aFence );

        VkSemaphore CreateVkSemaphore();
        void        DestroySemaphore( VkSemaphore aFence );

        VkDeviceMemory AllocateMemory( VkBuffer aVkBufferObject, size_t aSize, bool aIsHostVisible, bool aIsCudaShareable, size_t *aAllocatedSize = nullptr );
        VkDeviceMemory AllocateMemory( VkImage aVkImageObject, size_t aSize, bool aIsHostVisible, bool aIsCudaShareable, size_t *aAllocatedSize = nullptr );
        void           FreeMemory( VkDeviceMemory aMemory );

        template <typename _MapType>
        _MapType *MapMemory( VkDeviceMemory aMemory, size_t aSize, size_t aOffset )
        {
            if( aMemory == VK_NULL_HANDLE ) return nullptr;

            void *lMappedData;
            vkMapMemory( mVkLogicalDevice, aMemory, aOffset * sizeof( _MapType ), aSize * sizeof( _MapType ), 0, &lMappedData );
            return reinterpret_cast<_MapType *>( lMappedData );
        }
        void UnmapMemory( VkDeviceMemory aMemory ) { vkUnmapMemory( mVkLogicalDevice, aMemory ); }

        std::vector<VkCommandBuffer> AllocateCommandBuffer( uint32_t aCount );
        void                         DestroyCommandBuffer( VkCommandBuffer aBuffer );

        void BindMemory( VkBuffer aVkBufferObject, VkDeviceMemory aMemory );
        void BindMemory( VkImage aVkBufferObject, VkDeviceMemory aMemory );

        void *GetSharedMemoryHandle( VkDeviceMemory aVkMemory );

        VkQueue GetGraphicsQueue() { return mVkGraphicsQueue; }
        VkQueue GetPresentQueue() { return mVkPresentQueue; }

        VkFormat GetDepthFormat() { return mDepthFormat; }

        void UpdateDescriptorSets( VkWriteDescriptorSet aWriteOps );

        VkResult AcquireNextImage( VkSwapchainKHR aSwapChain, uint64_t aTimeout, VkSemaphore aWaitSemaphore,
                                   uint32_t *aNewImageIndex );
        VkResult Present( VkSwapchainKHR aSwapChain, uint32_t aImageIndex, VkSemaphore aWaitSemaphore );

        void WaitIdle();
        void WaitIdle( VkQueue aQueue );

      private:
        struct MemoryProperties
        {
            bool mHostVisible = false;
            bool mCudaVisible = false;
        };

        struct BufferProperties
        {
            bool mCudaVisible = false;
        };

        std::unordered_map<VkDeviceMemory, MemoryProperties> mMemoryPropertyLookup;
        std::unordered_map<VkBuffer, BufferProperties>       mBufferPropertyLookup;

        std::set<VkCommandBuffer>       mCommandBuffers;
        std::set<VkImage>               mImages;
        std::set<VkSampler>             mImageSamplers;
        std::set<VkImageView>           mImageViews;
        std::set<VkFramebuffer>         mFramebuffers;
        std::set<VkRenderPass>          mRenderPasses;
        std::set<VkPipeline>            mPipelines;
        std::set<VkFence>               mFences;
        std::set<VkSemaphore>           mSemaphores;
        std::set<VkShaderModule>        mShaderModules;
        std::set<VkDescriptorSetLayout> mDescriptorSetLayouts;
        std::set<VkDescriptorPool>      mDescriptorPools;
        std::set<VkPipelineLayout>      mPipelineLayouts;
        std::set<VkSwapchainKHR>        mSwapChains;

      private:
        Ref<IWindow> mWindow;

        VkInstance       mVkInstance            = VK_NULL_HANDLE;
        VkPhysicalDevice mVkPhysicalDevice      = VK_NULL_HANDLE;
        VkSurfaceKHR     mVkSurface             = VK_NULL_HANDLE;
        VkDevice         mVkLogicalDevice       = VK_NULL_HANDLE;
        VkQueue          mVkGraphicsQueue       = VK_NULL_HANDLE;
        VkQueue          mVkPresentQueue        = VK_NULL_HANDLE;
        VkQueue          mVkComputeQueue        = VK_NULL_HANDLE;
        VkCommandPool    mVkGraphicsCommandPool = VK_NULL_HANDLE;

        VkDebugUtilsMessengerEXT mDebugMessenger = VK_NULL_HANDLE;

        VkFormat mDepthFormat = VK_FORMAT_UNDEFINED;

      private:
        VkPhysicalDeviceProperties mPhysicalDeviceProperties;
        uint32_t                   mGraphicFamily = std::numeric_limits<uint32_t>::max();
        uint32_t                   mPresentFamily = std::numeric_limits<uint32_t>::max();
    };
} // namespace SE::Graphics::Internal