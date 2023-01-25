#include "VkGraphicContext.h"
#include "Graphics/Vulkan/VkCoreMacros.h"

#include <algorithm>
#include <set>
#include <string>

#include "VkTexture2D.h"

namespace SE::Graphics
{
    namespace
    {
        struct sSwapChainSupportDetails
        {
            VkSurfaceCapabilitiesKHR        mCapabilities;
            std::vector<VkSurfaceFormatKHR> mFormats;
            std::vector<VkPresentModeKHR>   mPresentModes;
        };

        std::vector<const char *> GetRequiredInstanceExtensions( bool aEnableValidationLayers )
        {
            uint32_t     lGlfwExtensionCount = 0;
            const char **lGlfwExtensions;
            lGlfwExtensions = glfwGetRequiredInstanceExtensions( &lGlfwExtensionCount );

            std::vector<const char *> lRequiredExtensions( lGlfwExtensions, lGlfwExtensions + lGlfwExtensionCount );
            lRequiredExtensions.push_back( VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME );
            lRequiredExtensions.push_back( VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME );
            lRequiredExtensions.push_back( VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME );

            if( aEnableValidationLayers ) lRequiredExtensions.push_back( VK_EXT_DEBUG_UTILS_EXTENSION_NAME );

            return lRequiredExtensions;
        }

        template <typename _Ty>
        bool IsSubset( std::set<_Ty> aA, std::set<_Ty> aB )
        {
            for( const auto &lX : aA )
                if( aB.find( lX ) == aB.end() )
                {
                    return false;
                }

            return true;
        }

        bool CheckValidationLayerSupport( const std::vector<const char *> aValidationLayers )
        {
            uint32_t lLayerCount;
            vkEnumerateInstanceLayerProperties( &lLayerCount, nullptr );

            std::vector<VkLayerProperties> lAvailableLayers( lLayerCount );
            vkEnumerateInstanceLayerProperties( &lLayerCount, lAvailableLayers.data() );

            std::set<std::string> lAvailableValidationLayers;
            for( const auto &lLayer : lAvailableLayers ) lAvailableValidationLayers.emplace( lLayer.layerName );

            std::set<std::string> lRequestedValidationLayers( aValidationLayers.begin(), aValidationLayers.end() );

            return IsSubset( lRequestedValidationLayers, lAvailableValidationLayers );
        }

        bool CkeckRequiredInstanceExtensions( const std::vector<const char *> aRequiredExtensions )
        {
            uint32_t lExtensionCount = 0;
            vkEnumerateInstanceExtensionProperties( nullptr, &lExtensionCount, nullptr );

            std::vector<VkExtensionProperties> lAvailableExtensions( lExtensionCount );
            vkEnumerateInstanceExtensionProperties( nullptr, &lExtensionCount, lAvailableExtensions.data() );

            std::set<const char *> lAvailableExtensionSet;
            for( const auto &lExtension : lAvailableExtensions ) lAvailableExtensionSet.emplace( lExtension.extensionName );

            std::set<const char *> lRequestedExtensionSet( aRequiredExtensions.begin(), aRequiredExtensions.end() );

            return IsSubset( lRequestedExtensionSet, lAvailableExtensionSet );
        }

        bool CheckRequiredDeviceExtensions( VkPhysicalDevice aVkPhysicalDevice, std::vector<const char *> aRequestedExtensions )
        {
            uint32_t lExtensionCount;
            vkEnumerateDeviceExtensionProperties( aVkPhysicalDevice, nullptr, &lExtensionCount, nullptr );

            std::vector<VkExtensionProperties> lAvailableExtensions( lExtensionCount );
            vkEnumerateDeviceExtensionProperties( aVkPhysicalDevice, nullptr, &lExtensionCount, lAvailableExtensions.data() );

            std::set<std::string> lRequestedExtensionsSet( aRequestedExtensions.begin(), aRequestedExtensions.end() );

            std::set<std::string> lAvailableExtensionSet;
            for( const auto &lExtension : lAvailableExtensions ) lAvailableExtensionSet.emplace( lExtension.extensionName );

            return IsSubset( lRequestedExtensionsSet, lAvailableExtensionSet );
        }

        std::tuple<uint32_t, uint32_t> GetQueueFamilies( VkPhysicalDevice aVkPhysicalDevice, VkSurfaceKHR aVkSurface )
        {
            uint32_t lQueueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties( aVkPhysicalDevice, &lQueueFamilyCount, nullptr );

            std::vector<VkQueueFamilyProperties> lAvailableQueueFamilies( lQueueFamilyCount );
            vkGetPhysicalDeviceQueueFamilyProperties( aVkPhysicalDevice, &lQueueFamilyCount, lAvailableQueueFamilies.data() );

            int lCurrentQueueIndex = 0;
            int lGraphicsFamily    = std::numeric_limits<uint32_t>::max();
            int lPresentFamily     = std::numeric_limits<uint32_t>::max();

            bool lGraphicsFamilyHasValue = false;
            bool lPresentFamilyHasValue  = false;
            for( const auto &lQueueFamily : lAvailableQueueFamilies )
            {
                if( lQueueFamily.queueCount > 0 && lQueueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT )
                {
                    lGraphicsFamily         = lCurrentQueueIndex;
                    lGraphicsFamilyHasValue = true;
                }

                VkBool32 lQueueSupportsPresentation = false;
                vkGetPhysicalDeviceSurfaceSupportKHR( aVkPhysicalDevice, lCurrentQueueIndex, aVkSurface, &lQueueSupportsPresentation );
                if( lQueueFamily.queueCount > 0 && lQueueSupportsPresentation )
                {
                    lPresentFamily         = lCurrentQueueIndex;
                    lPresentFamilyHasValue = true;
                }

                if( lGraphicsFamilyHasValue && lPresentFamilyHasValue ) break;

                lCurrentQueueIndex++;
            }

            return { lGraphicsFamily, lPresentFamily };
        }

        sSwapChainSupportDetails QuerySwapChainSupport( VkPhysicalDevice aVkPhysicalDevice, VkSurfaceKHR aVkSurface )
        {
            sSwapChainSupportDetails lSwapChainSupportDetails;
            vkGetPhysicalDeviceSurfaceCapabilitiesKHR( aVkPhysicalDevice, aVkSurface, &lSwapChainSupportDetails.mCapabilities );

            uint32_t lSwapchainFormatCount;
            vkGetPhysicalDeviceSurfaceFormatsKHR( aVkPhysicalDevice, aVkSurface, &lSwapchainFormatCount, nullptr );

            if( lSwapchainFormatCount != 0 )
            {
                lSwapChainSupportDetails.mFormats.resize( lSwapchainFormatCount );
                vkGetPhysicalDeviceSurfaceFormatsKHR( aVkPhysicalDevice, aVkSurface, &lSwapchainFormatCount,
                                                      lSwapChainSupportDetails.mFormats.data() );
            }

            uint32_t lSwapChainPresentModeCount;
            vkGetPhysicalDeviceSurfacePresentModesKHR( aVkPhysicalDevice, aVkSurface, &lSwapChainPresentModeCount, nullptr );

            if( lSwapChainPresentModeCount != 0 )
            {
                lSwapChainSupportDetails.mPresentModes.resize( lSwapChainPresentModeCount );
                vkGetPhysicalDeviceSurfacePresentModesKHR( aVkPhysicalDevice, aVkSurface, &lSwapChainPresentModeCount,
                                                           lSwapChainSupportDetails.mPresentModes.data() );
            }

            return lSwapChainSupportDetails;
        }

        bool DeviceIsSuitable( VkPhysicalDevice aVkPhysicalDevice, VkSurfaceKHR aVkSurface,
                               std::vector<const char *> aRequestedExtensions )
        {
            bool lRequiredExtensionsAreSupported = CheckRequiredDeviceExtensions( aVkPhysicalDevice, aRequestedExtensions );

            bool lSwapChainIsAdequate = false;
            if( lRequiredExtensionsAreSupported )
            {
                sSwapChainSupportDetails lDetails = QuerySwapChainSupport( aVkPhysicalDevice, aVkSurface );
                lSwapChainIsAdequate              = !lDetails.mFormats.empty() && !lDetails.mPresentModes.empty();
            }

            VkPhysicalDeviceFeatures lSupportedPhysicalDeviceFeatures;
            vkGetPhysicalDeviceFeatures( aVkPhysicalDevice, &lSupportedPhysicalDeviceFeatures );

            VkPhysicalDeviceDescriptorIndexingFeaturesEXT lIndexingFeatures{};
            lIndexingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT;
            lIndexingFeatures.pNext = nullptr;

            VkPhysicalDeviceFeatures2 lDeviceFeatures2{};
            lDeviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            lDeviceFeatures2.pNext = &lIndexingFeatures;
            vkGetPhysicalDeviceFeatures2( aVkPhysicalDevice, &lDeviceFeatures2 );

            auto [GraphicsFamily, PresentFamily] = GetQueueFamilies( aVkPhysicalDevice, aVkSurface );

            return ( lIndexingFeatures.descriptorBindingPartiallyBound ) && ( lIndexingFeatures.runtimeDescriptorArray ) &&
                   ( GraphicsFamily != 0xffffff ) && ( PresentFamily != 0xffffff ) && lSwapChainIsAdequate &&
                   lSupportedPhysicalDeviceFeatures.samplerAnisotropy;
        }

        std::vector<VkPhysicalDevice> EnumeratePhysicalDevices( VkInstance aVkInstance, VkSurfaceKHR aVkSurface )
        {
            uint32_t lPhysicalDeviceCount = 0;
            vkEnumeratePhysicalDevices( aVkInstance, &lPhysicalDeviceCount, nullptr );

            std::vector<VkPhysicalDevice> lPhysicalDeviceObjects( lPhysicalDeviceCount );
            if( 0 != lPhysicalDeviceCount )
                vkEnumeratePhysicalDevices( aVkInstance, &lPhysicalDeviceCount, lPhysicalDeviceObjects.data() );

            return lPhysicalDeviceObjects;
        }

        VkPhysicalDevice PickPhysicalDevice( VkInstance aVkInstance, VkSurfaceKHR aVkSurface,
                                             std::vector<const char *> aRequestedExtensions )
        {
            std::vector<VkPhysicalDevice> lPhysicalDeviceList = EnumeratePhysicalDevices( aVkInstance, aVkSurface );

            for( const auto &lPhysicalDevice : lPhysicalDeviceList )
            {
                if( DeviceIsSuitable( lPhysicalDevice, aVkSurface, aRequestedExtensions ) )
                {
                    return lPhysicalDevice;
                }
            }

            throw std::runtime_error( "failed to find a suitable GPU!" );
        }

        VkFormat FindSupportedFormat( VkPhysicalDevice aVkPhysicalDevice, const std::vector<VkFormat> &aCandidates,
                                      VkImageTiling aImageTiling, VkFormatFeatureFlags aFeatures )
        {
            for( VkFormat lCurrentFormat : aCandidates )
            {
                VkFormatProperties lProperties;
                vkGetPhysicalDeviceFormatProperties( aVkPhysicalDevice, lCurrentFormat, &lProperties );

                if( aImageTiling == VK_IMAGE_TILING_LINEAR && ( lProperties.linearTilingFeatures & aFeatures ) == aFeatures )
                    return lCurrentFormat;
                else if( aImageTiling == VK_IMAGE_TILING_OPTIMAL && ( lProperties.optimalTilingFeatures & aFeatures ) == aFeatures )
                    return lCurrentFormat;
            }

            throw std::runtime_error( "failed to find supported format!" );
        }

        VkFormat FindDepthFormat( VkPhysicalDevice aVkPhysicalDevice )
        {
            return FindSupportedFormat( aVkPhysicalDevice,
                                        { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
                                        VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT );
        }

        uint32_t FindMemoryTypeIndex( VkPhysicalDevice aVkPhysicalDevice, uint32_t aTypeFilter,
                                      VkMemoryPropertyFlags aMemoryProperties )
        {
            VkPhysicalDeviceMemoryProperties lMemoryProperties;
            vkGetPhysicalDeviceMemoryProperties( aVkPhysicalDevice, &lMemoryProperties );

            for( uint32_t i = 0; i < lMemoryProperties.memoryTypeCount; i++ )
            {
                if( ( aTypeFilter & ( 1 << i ) ) &&
                    ( lMemoryProperties.memoryTypes[i].propertyFlags & aMemoryProperties ) == aMemoryProperties )
                    return i;
            }

            throw std::runtime_error( "failed to find suitable memory type!" );
        }

        VkDeviceMemory DoAllocateMemory( VkPhysicalDevice aVkPhysicalDevice, VkDevice aVkLogicalDevice,
                                         VkMemoryRequirements aMemoryRequirements, size_t aSize, bool aIsHostVisible,
                                         bool aIsCudaShareable, bool aIsImage )
        {
            VkDeviceMemory lNewMemory;

            VkMemoryAllocateInfo lMemoryAllocationIfo{};
            lMemoryAllocationIfo.sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            lMemoryAllocationIfo.allocationSize = aMemoryRequirements.size;

            VkMemoryPropertyFlags lMemoryProperties{};
            lMemoryProperties = aIsHostVisible ? ( VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT )
                                               : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

            lMemoryAllocationIfo.memoryTypeIndex =
                FindMemoryTypeIndex( aVkPhysicalDevice, aMemoryRequirements.memoryTypeBits, lMemoryProperties );

            VkExportMemoryAllocateInfoKHR lVulkanExportMemoryAllocateInfoKHR{};
            if( aIsCudaShareable )
            {
                lVulkanExportMemoryAllocateInfoKHR.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
                lVulkanExportMemoryAllocateInfoKHR.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
                lVulkanExportMemoryAllocateInfoKHR.pNext       = NULL;
                lMemoryAllocationIfo.pNext                     = &lVulkanExportMemoryAllocateInfoKHR;
            }

            VK_CHECK_RESULT( vkAllocateMemory( aVkLogicalDevice, &lMemoryAllocationIfo, nullptr, &lNewMemory ) );

            return lNewMemory;
        }

        VkSurfaceFormatKHR ChooseSwapSurfaceFormat( const std::vector<VkSurfaceFormatKHR> &aAvailableFormats )
        {
            for( const auto &lFormat : aAvailableFormats )
            {
                if( lFormat.format == VK_FORMAT_B8G8R8A8_SRGB && lFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR )
                    return lFormat;
            }

            return aAvailableFormats[0];
        }

        VkPresentModeKHR ChooseSwapPresentMode( const std::vector<VkPresentModeKHR> &aAvailablePresentModes )
        {
            for( const auto &lPresentMode : aAvailablePresentModes )
            {
                if( lPresentMode == VK_PRESENT_MODE_MAILBOX_KHR )
                {
                    std::cout << "Present mode: Mailbox" << std::endl;
                    return lPresentMode;
                }
            }

            std::cout << "Present mode: V-Sync" << std::endl;
            return VK_PRESENT_MODE_FIFO_KHR;
        }

        VkExtent2D ChooseSwapExtent( VkExtent2D mVkSurfaceExtent, const VkSurfaceCapabilitiesKHR &aCapabilities )
        {
            if( aCapabilities.currentExtent.width != std::numeric_limits<uint32_t>::max() )
            {
                return aCapabilities.currentExtent;
            }
            else
            {
                VkExtent2D lActualExtent = mVkSurfaceExtent;
                lActualExtent.width      = std::max( aCapabilities.minImageExtent.width,
                                                     std::min( aCapabilities.maxImageExtent.width, lActualExtent.width ) );
                lActualExtent.height     = std::max( aCapabilities.minImageExtent.height,
                                                     std::min( aCapabilities.maxImageExtent.height, lActualExtent.height ) );

                return lActualExtent;
            }
        }

        VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback( VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
                                                      VkDebugUtilsMessageTypeFlagsEXT             messageType,
                                                      const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData )
        {
            SE::Logging::Error( pCallbackData->pMessage );
            return VK_FALSE;
        }
    } // namespace

    VkGraphicContext::VkGraphicContext( Ref<Core::IWindow> aWindow, uint32_t aSampleCount, bool aEnableValidation )
        : IGraphicContext( aWindow, aSampleCount )
    {
        VkApplicationInfo lApplicationInfo{};
        lApplicationInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        lApplicationInfo.pApplicationName   = "";
        lApplicationInfo.applicationVersion = VK_MAKE_VERSION( 1, 0, 0 );
        lApplicationInfo.pEngineName        = "";
        lApplicationInfo.engineVersion      = VK_MAKE_VERSION( 1, 0, 0 );
        lApplicationInfo.apiVersion         = VK_API_VERSION_1_0;

        VkInstanceCreateInfo lInstanceCreateInfo{};
        lInstanceCreateInfo.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        lInstanceCreateInfo.pApplicationInfo = &lApplicationInfo;

        auto lRequiredExtensions                    = GetRequiredInstanceExtensions( aEnableValidation );
        lInstanceCreateInfo.enabledExtensionCount   = static_cast<uint32_t>( lRequiredExtensions.size() );
        lInstanceCreateInfo.ppEnabledExtensionNames = lRequiredExtensions.data();

        const std::vector<const char *> lRequiredValidationLayers = { "VK_LAYER_KHRONOS_validation" };

        bool                               lShouldAttatchDebugCallback = false;
        VkDebugUtilsMessengerCreateInfoEXT lDebugOutputCreateInfo{};
        if( aEnableValidation && CheckValidationLayerSupport( lRequiredValidationLayers ) )
        {
            lInstanceCreateInfo.enabledLayerCount   = static_cast<uint32_t>( lRequiredValidationLayers.size() );
            lInstanceCreateInfo.ppEnabledLayerNames = lRequiredValidationLayers.data();

            lDebugOutputCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            lDebugOutputCreateInfo.messageSeverity =
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            lDebugOutputCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                                 VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                                 VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            lDebugOutputCreateInfo.pfnUserCallback = debugCallback;
            lDebugOutputCreateInfo.pUserData       = nullptr;
            lInstanceCreateInfo.pNext              = (VkDebugUtilsMessengerCreateInfoEXT *)&lDebugOutputCreateInfo;
            lShouldAttatchDebugCallback            = true;
        }
        else
        {
            lInstanceCreateInfo.enabledLayerCount = 0;
            lInstanceCreateInfo.pNext             = nullptr;
        }

        VK_CHECK_RESULT( vkCreateInstance( &lInstanceCreateInfo, nullptr, &mVkInstance ) );

        if( lShouldAttatchDebugCallback )
        {
            auto lVkCreateDebugUtilsMessengerEXT =
                (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr( mVkInstance, "vkCreateDebugUtilsMessengerEXT" );

            if( lVkCreateDebugUtilsMessengerEXT != nullptr )
                VK_CHECK_RESULT( lVkCreateDebugUtilsMessengerEXT( mVkInstance, &lDebugOutputCreateInfo, nullptr, &mDebugMessenger ) );
            else
                VK_CHECK_RESULT( VK_ERROR_EXTENSION_NOT_PRESENT );
        }

        VK_CHECK_RESULT( glfwCreateWindowSurface( mVkInstance, aWindow->GetGLFWWindow(), nullptr, &mVkSurface ) );

        const std::vector<const char *> lLogicalDeviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                                                                     VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
                                                                     VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
                                                                     VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
                                                                     VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
                                                                     VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME };

        mVkPhysicalDevice = PickPhysicalDevice( mVkInstance, mVkSurface, lLogicalDeviceExtensions );
        mDepthFormat      = FindDepthFormat( mVkPhysicalDevice );
        vkGetPhysicalDeviceProperties( mVkPhysicalDevice, &mPhysicalDeviceProperties );

        std::tie( mGraphicFamily, mPresentFamily ) = GetQueueFamilies( mVkPhysicalDevice, mVkSurface );

        std::vector<VkDeviceQueueCreateInfo> lLogicalDeviceQueueCreateInfos;
        std::set<uint32_t>                   lUniqueQueueFamilies = { mGraphicFamily, mPresentFamily };

        float lQueuePriority = 1.0f;
        for( uint32_t lQueueFamily : lUniqueQueueFamilies )
        {
            VkDeviceQueueCreateInfo lQueueCreateInfo{};
            lQueueCreateInfo.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            lQueueCreateInfo.queueFamilyIndex = lQueueFamily;
            lQueueCreateInfo.queueCount       = 1;
            lQueueCreateInfo.pQueuePriorities = &lQueuePriority;
            lLogicalDeviceQueueCreateInfos.push_back( lQueueCreateInfo );
        }

        VkPhysicalDeviceDescriptorIndexingFeaturesEXT lLogicalDeviceIndexingFeatures{};
        lLogicalDeviceIndexingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT;
        lLogicalDeviceIndexingFeatures.pNext = nullptr;
        lLogicalDeviceIndexingFeatures.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
        lLogicalDeviceIndexingFeatures.descriptorBindingVariableDescriptorCount  = VK_TRUE;
        lLogicalDeviceIndexingFeatures.descriptorBindingPartiallyBound           = VK_TRUE;
        lLogicalDeviceIndexingFeatures.runtimeDescriptorArray                    = VK_TRUE;

        VkPhysicalDeviceFeatures lPhysicalDeviceFeatures{};
        lPhysicalDeviceFeatures.samplerAnisotropy = VK_TRUE;
        lPhysicalDeviceFeatures.fillModeNonSolid  = VK_TRUE;
        lPhysicalDeviceFeatures.wideLines         = VK_TRUE;

        VkDeviceCreateInfo lLogicalDeviceCreateInfo{};
        lLogicalDeviceCreateInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        lLogicalDeviceCreateInfo.queueCreateInfoCount    = static_cast<uint32_t>( lLogicalDeviceQueueCreateInfos.size() );
        lLogicalDeviceCreateInfo.pQueueCreateInfos       = lLogicalDeviceQueueCreateInfos.data();
        lLogicalDeviceCreateInfo.pEnabledFeatures        = &lPhysicalDeviceFeatures;
        lLogicalDeviceCreateInfo.enabledExtensionCount   = static_cast<uint32_t>( lLogicalDeviceExtensions.size() );
        lLogicalDeviceCreateInfo.ppEnabledExtensionNames = lLogicalDeviceExtensions.data();
        lLogicalDeviceCreateInfo.pNext                   = &lLogicalDeviceIndexingFeatures;

        if( lShouldAttatchDebugCallback )
        {
            lLogicalDeviceCreateInfo.enabledLayerCount   = static_cast<uint32_t>( lRequiredValidationLayers.size() );
            lLogicalDeviceCreateInfo.ppEnabledLayerNames = lRequiredValidationLayers.data();
        }
        else
        {
            lLogicalDeviceCreateInfo.enabledLayerCount = 0;
        }

        VK_CHECK_RESULT( vkCreateDevice( mVkPhysicalDevice, &lLogicalDeviceCreateInfo, nullptr, &mVkLogicalDevice ) );

        vkGetDeviceQueue( mVkLogicalDevice, mGraphicFamily, 0, &mVkGraphicsQueue );
        vkGetDeviceQueue( mVkLogicalDevice, mPresentFamily, 0, &mVkPresentQueue );

        VkCommandPoolCreateInfo lGraphicsCommandPoolCreateInfo{};
        lGraphicsCommandPoolCreateInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        lGraphicsCommandPoolCreateInfo.queueFamilyIndex = mGraphicFamily;
        lGraphicsCommandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CHECK_RESULT( vkCreateCommandPool( mVkLogicalDevice, &lGraphicsCommandPoolCreateInfo, nullptr, &mVkGraphicsCommandPool ) );

        std::vector<VkDescriptorPoolSize> lPoolSizes( 4 );
        lPoolSizes[0]     = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10000 };
        lPoolSizes[1]     = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 10000 };
        lPoolSizes[2]     = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10000 };
        lPoolSizes[3]     = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10000 };
        mVkDescriptorPool = CreateDescriptorPool( 10000, lPoolSizes );
    }

    VkGraphicContext::~VkGraphicContext()
    {
        auto lVkDestroyDebugUtilsMessengerEXT =
            (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr( mVkInstance, "vkDestroyDebugUtilsMessengerEXT" );
        if( lVkDestroyDebugUtilsMessengerEXT != nullptr ) lVkDestroyDebugUtilsMessengerEXT( mVkInstance, mDebugMessenger, nullptr );

        for( auto &lSemaphore : mSemaphores ) vkDestroySemaphore( mVkLogicalDevice, lSemaphore, nullptr );

        for( auto &lFence : mFences ) vkDestroyFence( mVkLogicalDevice, lFence, nullptr );

        for( auto &lSwapChain : mSwapChains ) vkDestroySwapchainKHR( mVkLogicalDevice, lSwapChain, nullptr );

        for( auto &lDescriptorPool : mDescriptorPools ) vkDestroyDescriptorPool( mVkLogicalDevice, lDescriptorPool, nullptr );

        for( auto &lPipeline : mPipelines ) vkDestroyPipeline( mVkLogicalDevice, lPipeline, nullptr );

        for( auto &lDescriptorSetLayout : mDescriptorSetLayouts )
            vkDestroyDescriptorSetLayout( mVkLogicalDevice, lDescriptorSetLayout, nullptr );

        for( auto &lShaderModule : mShaderModules ) vkDestroyShaderModule( mVkLogicalDevice, lShaderModule, nullptr );

        for( auto &lRenderPass : mRenderPasses ) vkDestroyRenderPass( mVkLogicalDevice, lRenderPass, nullptr );

        for( auto &lFramebuffer : mFramebuffers ) vkDestroyFramebuffer( mVkLogicalDevice, lFramebuffer, nullptr );

        for( auto &lMemory : mMemoryPropertyLookup ) vkFreeMemory( mVkLogicalDevice, lMemory.first, nullptr );

        for( auto &lBuffer : mBufferPropertyLookup ) vkDestroyBuffer( mVkLogicalDevice, lBuffer.first, nullptr );

        for( auto &lImageSampler : mImageSamplers ) vkDestroySampler( mVkLogicalDevice, lImageSampler, nullptr );

        for( auto &lImage : mImages ) vkDestroyImage( mVkLogicalDevice, lImage, nullptr );

        if( mVkGraphicsCommandPool != VK_NULL_HANDLE ) vkDestroyCommandPool( mVkLogicalDevice, mVkGraphicsCommandPool, nullptr );

        if( mVkLogicalDevice != VK_NULL_HANDLE ) vkDestroyDevice( mVkLogicalDevice, nullptr );

        if( mVkSurface != VK_NULL_HANDLE ) vkDestroySurfaceKHR( mVkInstance, mVkSurface, nullptr );

        if( mVkInstance != VK_NULL_HANDLE ) vkDestroyInstance( mVkInstance, nullptr );
    }

    VkDeviceMemory VkGraphicContext::AllocateMemory( VkImage aVkImageObject, size_t aSize, bool aIsHostVisible, bool aIsCudaShareable,
                                                     size_t *aAllocatedSize )
    {
        VkMemoryRequirements lMemoryRequirements;
        vkGetImageMemoryRequirements( mVkLogicalDevice, aVkImageObject, &lMemoryRequirements );

        VkDeviceMemory lNewMemory         = DoAllocateMemory( mVkPhysicalDevice, mVkLogicalDevice, lMemoryRequirements,
                                                              lMemoryRequirements.size, aIsHostVisible, aIsCudaShareable, true );
        mMemoryPropertyLookup[lNewMemory] = MemoryProperties{ aIsHostVisible, aIsCudaShareable };
        if( aAllocatedSize ) *aAllocatedSize = lMemoryRequirements.size;
        return lNewMemory;
    }

    VkDeviceMemory VkGraphicContext::AllocateMemory( VkBuffer aVkBufferObject, size_t aSize, bool aIsHostVisible,
                                                     bool aIsCudaShareable, size_t *aAllocatedSize )
    {
        VkMemoryRequirements lMemoryRequirements;
        vkGetBufferMemoryRequirements( mVkLogicalDevice, aVkBufferObject, &lMemoryRequirements );

        VkDeviceMemory lNewMemory = DoAllocateMemory( mVkPhysicalDevice, mVkLogicalDevice, lMemoryRequirements, aSize, aIsHostVisible,
                                                      aIsCudaShareable, false );
        mMemoryPropertyLookup[lNewMemory] = MemoryProperties{ aIsHostVisible, aIsCudaShareable };
        if( aAllocatedSize ) *aAllocatedSize = lMemoryRequirements.size;
        return lNewMemory;
    }

    void VkGraphicContext::FreeMemory( VkDeviceMemory aMemory )
    {
        if( mMemoryPropertyLookup.find( aMemory ) != mMemoryPropertyLookup.end() )
        {
            mMemoryPropertyLookup.erase( aMemory );

            if( aMemory == VK_NULL_HANDLE ) return;

            vkFreeMemory( mVkLogicalDevice, aMemory, nullptr );
        }
    }

    void *VkGraphicContext::GetSharedMemoryHandle( VkDeviceMemory aVkMemory )
    {
        if( aVkMemory == VK_NULL_HANDLE ) return nullptr;

        if( mMemoryPropertyLookup.find( aVkMemory ) == mMemoryPropertyLookup.end() ) return nullptr;

        if( !mMemoryPropertyLookup[aVkMemory].mCudaVisible ) return nullptr;

        VkMemoryGetWin32HandleInfoKHR lVkMemoryGetWin32HandleInfoKHR{};
        lVkMemoryGetWin32HandleInfoKHR.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
        lVkMemoryGetWin32HandleInfoKHR.pNext      = NULL;
        lVkMemoryGetWin32HandleInfoKHR.memory     = aVkMemory;
        lVkMemoryGetWin32HandleInfoKHR.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

        PFN_vkGetMemoryWin32HandleKHR lVkGetMemoryWin32HandleKHR;
        lVkGetMemoryWin32HandleKHR =
            (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr( mVkLogicalDevice, "vkGetMemoryWin32HandleKHR" );
        if( !lVkGetMemoryWin32HandleKHR ) return nullptr;

        void *lSharedMemoryHandle;
        auto lResult = lVkGetMemoryWin32HandleKHR( mVkLogicalDevice, &lVkMemoryGetWin32HandleInfoKHR, (HANDLE *)&lSharedMemoryHandle );

        if( lResult != VK_SUCCESS ) return nullptr;

        return lSharedMemoryHandle;
    }

    void VkGraphicContext::BindMemory( VkBuffer aVkBufferObject, VkDeviceMemory aMemory )
    {
        VK_CHECK_RESULT( vkBindBufferMemory( mVkLogicalDevice, aVkBufferObject, aMemory, 0 ) );
    }

    void VkGraphicContext::BindMemory( VkImage aVkImageObject, VkDeviceMemory aMemory )
    {
        VK_CHECK_RESULT( vkBindImageMemory( mVkLogicalDevice, aVkImageObject, aMemory, 0 ) );
    }

    std::vector<VkCommandBuffer> VkGraphicContext::AllocateCommandBuffer( uint32_t aCount )
    {
        std::vector<VkCommandBuffer> lNewCommandBuffers( aCount );

        VkCommandBufferAllocateInfo lCommandBufferAllocInfo{};
        lCommandBufferAllocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        lCommandBufferAllocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        lCommandBufferAllocInfo.commandPool        = mVkGraphicsCommandPool;
        lCommandBufferAllocInfo.commandBufferCount = aCount;

        VK_CHECK_RESULT( vkAllocateCommandBuffers( mVkLogicalDevice, &lCommandBufferAllocInfo, lNewCommandBuffers.data() ) );

        for( auto &lX : lNewCommandBuffers ) mCommandBuffers.emplace( lX );
        return std::move( lNewCommandBuffers );
    }

    void VkGraphicContext::DestroyCommandBuffer( VkCommandBuffer aBuffer )
    {
        if( mCommandBuffers.find( aBuffer ) != mCommandBuffers.end() )
        {
            mCommandBuffers.erase( aBuffer );
            vkFreeCommandBuffers( mVkLogicalDevice, mVkGraphicsCommandPool, 1, &aBuffer );
        }
    }

    VkBuffer VkGraphicContext::CreateBuffer( VkBufferUsageFlags aBufferFlags, size_t aSize, bool aIsHostVisible,
                                             bool aIsCudaShareable )
    {
        VkBufferCreateInfo lBufferCreateInfo{};
        lBufferCreateInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        lBufferCreateInfo.size        = aSize;
        lBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        lBufferCreateInfo.usage       = aBufferFlags;

        VkExternalMemoryBufferCreateInfo lExternalMemoryBufferInfo{};
        if( aIsCudaShareable )
        {
            lExternalMemoryBufferInfo.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
            lExternalMemoryBufferInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
            lBufferCreateInfo.pNext               = &lExternalMemoryBufferInfo;
        }

        VkBuffer lNewBuffer{};
        VK_CHECK_RESULT( vkCreateBuffer( mVkLogicalDevice, &lBufferCreateInfo, nullptr, &lNewBuffer ) );
        mBufferPropertyLookup[lNewBuffer] = BufferProperties{ aIsHostVisible };

        return lNewBuffer;
    }

    void VkGraphicContext::DestroyBuffer( VkBuffer aBuffer )
    {

        if( mBufferPropertyLookup.find( aBuffer ) != mBufferPropertyLookup.end() )
        {
            mBufferPropertyLookup.erase( aBuffer );
            vkDestroyBuffer( mVkLogicalDevice, aBuffer, nullptr );
        }
    }

    VkImage VkGraphicContext::CreateImage( uint32_t aWidth, uint32_t aHeight, uint32_t aDepth, uint32_t aMipLevels, uint32_t aLayers,
                                           uint8_t aSampleCount, bool aIsCudaShareable, bool aCubeCompatible, VkFormat aFormat,
                                           VkMemoryPropertyFlags aProperties, VkImageUsageFlags aUsage )
    {
        VkImageCreateInfo lImageCreateInfo{};
        lImageCreateInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        lImageCreateInfo.imageType     = VK_IMAGE_TYPE_2D;
        lImageCreateInfo.extent.width  = aWidth;
        lImageCreateInfo.extent.height = aHeight;
        lImageCreateInfo.extent.depth  = aDepth;
        lImageCreateInfo.mipLevels     = aMipLevels;
        lImageCreateInfo.arrayLayers   = aLayers;
        lImageCreateInfo.format        = aFormat;
        lImageCreateInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
        lImageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        lImageCreateInfo.usage         = aUsage;
        lImageCreateInfo.samples       = VK_SAMPLE_COUNT_VALUE( aSampleCount );
        lImageCreateInfo.flags         = aCubeCompatible ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;

        VkExternalMemoryImageCreateInfo lExternalMemoryBufferInfo{};
        if( aIsCudaShareable )
        {
            lExternalMemoryBufferInfo.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
            lExternalMemoryBufferInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
            lImageCreateInfo.pNext                = &lExternalMemoryBufferInfo;
        }

        VkImage lNewImage{};
        VK_CHECK_RESULT( vkCreateImage( mVkLogicalDevice, &lImageCreateInfo, nullptr, &lNewImage ) );
        mImages.emplace( lNewImage );

        return lNewImage;
    }

    void VkGraphicContext::DestroyImage( VkImage aImage )
    {
        if( mImages.find( aImage ) != mImages.end() )
        {
            mImages.erase( aImage );
            vkDestroyImage( mVkLogicalDevice, aImage, nullptr );
        }
    }

    VkSampler VkGraphicContext::CreateSampler( VkFilter aMinificationFilter, VkFilter aMagnificationFilter,
                                               VkSamplerAddressMode aWrappingMode, VkSamplerMipmapMode aMipmapMode )
    {
        VkSamplerCreateInfo lSamplerCreateInfo{};
        lSamplerCreateInfo.sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        lSamplerCreateInfo.magFilter               = aMinificationFilter;
        lSamplerCreateInfo.minFilter               = aMagnificationFilter;
        lSamplerCreateInfo.addressModeU            = aWrappingMode;
        lSamplerCreateInfo.addressModeV            = aWrappingMode;
        lSamplerCreateInfo.addressModeW            = aWrappingMode;
        lSamplerCreateInfo.anisotropyEnable        = VK_TRUE;
        lSamplerCreateInfo.maxAnisotropy           = mPhysicalDeviceProperties.limits.maxSamplerAnisotropy;
        lSamplerCreateInfo.borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        lSamplerCreateInfo.unnormalizedCoordinates = VK_FALSE;
        lSamplerCreateInfo.compareEnable           = VK_FALSE;
        lSamplerCreateInfo.compareOp               = VK_COMPARE_OP_ALWAYS;
        lSamplerCreateInfo.mipmapMode              = aMipmapMode;
        lSamplerCreateInfo.mipLodBias              = 0.0f;
        lSamplerCreateInfo.minLod                  = -1000;
        lSamplerCreateInfo.maxLod                  = 1000;

        VkSampler lNewSampler{};
        VK_CHECK_RESULT( vkCreateSampler( mVkLogicalDevice, &lSamplerCreateInfo, nullptr, &lNewSampler ) );
        mImageSamplers.emplace( lNewSampler );

        return lNewSampler;
    }

    void VkGraphicContext::DestroySampler( VkSampler aSampler )
    {
        if( mImageSamplers.find( aSampler ) != mImageSamplers.end() )
        {
            mImageSamplers.erase( aSampler );
            vkDestroySampler( mVkLogicalDevice, aSampler, nullptr );
        }
    }

    VkImageView VkGraphicContext::CreateImageView( VkImage aImageObject, uint32_t aLayerCount, VkImageViewType aViewType,
                                                   VkFormat aImageFormat, VkImageAspectFlags aAspectMask,
                                                   VkComponentMapping aComponentSwizzle )
    {
        return CreateImageView( aImageObject, 0, aLayerCount, aViewType, aImageFormat, aAspectMask, aComponentSwizzle );
    }

    VkImageView VkGraphicContext::CreateImageView( VkImage aImageObject, uint32_t aBaseLayer, uint32_t aLayerCount,
                                                   VkImageViewType aViewType, VkFormat aImageFormat, VkImageAspectFlags aAspectMask,
                                                   VkComponentMapping aComponentSwizzle )
    {
        VkImageViewCreateInfo lImageViewCreateInfo{};
        lImageViewCreateInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        lImageViewCreateInfo.image                           = aImageObject;
        lImageViewCreateInfo.viewType                        = aViewType;
        lImageViewCreateInfo.format                          = aImageFormat;
        lImageViewCreateInfo.subresourceRange.aspectMask     = aAspectMask;
        lImageViewCreateInfo.subresourceRange.baseMipLevel   = 0;
        lImageViewCreateInfo.subresourceRange.levelCount     = 1;
        lImageViewCreateInfo.subresourceRange.baseArrayLayer = aBaseLayer;
        lImageViewCreateInfo.subresourceRange.layerCount     = aLayerCount;
        lImageViewCreateInfo.components                      = aComponentSwizzle;

        VkImageView lNewImageView{};
        VK_CHECK_RESULT( vkCreateImageView( mVkLogicalDevice, &lImageViewCreateInfo, nullptr, &lNewImageView ) );
        mImageViews.emplace( lNewImageView );

        return lNewImageView;
    }

    void VkGraphicContext::DestroyImageView( VkImageView aImageView )
    {
        if( mImageViews.find( aImageView ) != mImageViews.end() )
        {
            mImageViews.erase( aImageView );
            vkDestroyImageView( mVkLogicalDevice, aImageView, nullptr );
        }
    }

    VkFramebuffer VkGraphicContext::CreateFramebuffer( std::vector<VkImageView> aImageViews, uint32_t aWidth, uint32_t aHeight,
                                                       uint32_t aLayers, VkRenderPass aRenderPass )
    {
        VkFramebufferCreateInfo lFramebufferCreateInfo{};
        lFramebufferCreateInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        lFramebufferCreateInfo.renderPass      = aRenderPass;
        lFramebufferCreateInfo.attachmentCount = static_cast<uint32_t>( aImageViews.size() );
        lFramebufferCreateInfo.pAttachments    = aImageViews.data();
        lFramebufferCreateInfo.width           = aWidth;
        lFramebufferCreateInfo.height          = aHeight;
        lFramebufferCreateInfo.layers          = aLayers;
        lFramebufferCreateInfo.pNext           = nullptr;
        lFramebufferCreateInfo.flags           = 0;

        VkFramebuffer lNewFramebuffer{};
        VK_CHECK_RESULT( vkCreateFramebuffer( mVkLogicalDevice, &lFramebufferCreateInfo, nullptr, &lNewFramebuffer ) );
        mFramebuffers.emplace( lNewFramebuffer );

        return lNewFramebuffer;
    }

    void VkGraphicContext::DestroyFramebuffer( VkFramebuffer aFramebuffer )
    {
        if( mFramebuffers.find( aFramebuffer ) != mFramebuffers.end() )
        {
            mFramebuffers.erase( aFramebuffer );
            vkDestroyFramebuffer( mVkLogicalDevice, aFramebuffer, nullptr );
        }
    }

    VkRenderPass VkGraphicContext::CreateRenderPass( std::vector<VkAttachmentDescription> aAttachments,
                                                     std::vector<VkSubpassDescription>    aSubpasses,
                                                     std::vector<VkSubpassDependency>     aSubpassDependencies )
    {
        VkRenderPassCreateInfo lRenderPassCreateInfo{};
        lRenderPassCreateInfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        lRenderPassCreateInfo.attachmentCount = static_cast<uint32_t>( aAttachments.size() );
        lRenderPassCreateInfo.pAttachments    = aAttachments.data();
        lRenderPassCreateInfo.subpassCount    = aSubpasses.size();
        lRenderPassCreateInfo.pSubpasses      = aSubpasses.data();
        lRenderPassCreateInfo.dependencyCount = aSubpassDependencies.size();
        lRenderPassCreateInfo.pDependencies   = aSubpassDependencies.data();

        VkRenderPass lNewRenderPass{};
        VK_CHECK_RESULT( vkCreateRenderPass( mVkLogicalDevice, &lRenderPassCreateInfo, nullptr, &lNewRenderPass ) );
        mRenderPasses.emplace( lNewRenderPass );

        return lNewRenderPass;
    }

    void VkGraphicContext::DestroyRenderPass( VkRenderPass aRenderPass )
    {
        if( mRenderPasses.find( aRenderPass ) != mRenderPasses.end() )
        {
            mRenderPasses.erase( aRenderPass );
            vkDestroyRenderPass( mVkLogicalDevice, aRenderPass, nullptr );
        }
    }

    VkPipeline VkGraphicContext::CreatePipeline( VkGraphicsPipelineCreateInfo aCreateInfo )
    {
        VkPipeline lNewPipeline;
        if( vkCreateGraphicsPipelines( mVkLogicalDevice, VK_NULL_HANDLE, 1, &aCreateInfo, nullptr, &lNewPipeline ) != VK_SUCCESS )
        {
            return VK_NULL_HANDLE;
        }
        mPipelines.emplace( lNewPipeline );

        return lNewPipeline;
    }

    void VkGraphicContext::DestroyPipeline( VkPipeline aFramebuffer )
    {
        if( mPipelines.find( aFramebuffer ) != mPipelines.end() )
        {
            mPipelines.erase( aFramebuffer );
            vkDestroyPipeline( mVkLogicalDevice, aFramebuffer, nullptr );
        }
    }

    VkFence VkGraphicContext::CreateFence()
    {
        VkFenceCreateInfo lFenceCreateInfo{};
        lFenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        lFenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        VkFence lNewFence{};
        VK_CHECK_RESULT( vkCreateFence( mVkLogicalDevice, &lFenceCreateInfo, nullptr, &lNewFence ) );
        mFences.emplace( lNewFence );

        return lNewFence;
    }

    void VkGraphicContext::DestroyFence( VkFence aFence )
    {
        if( mFences.find( aFence ) != mFences.end() )
        {
            mFences.erase( aFence );
            vkDestroyFence( mVkLogicalDevice, aFence, nullptr );
        }
    }

    VkSemaphore VkGraphicContext::CreateVkSemaphore()
    {
        VkSemaphoreCreateInfo lSemaphoreCreateInfo{};
        lSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkSemaphore lNewSemaphore{};
        VK_CHECK_RESULT( vkCreateSemaphore( mVkLogicalDevice, &lSemaphoreCreateInfo, nullptr, &lNewSemaphore ) );
        mSemaphores.emplace( lNewSemaphore );

        return lNewSemaphore;
    }

    void VkGraphicContext::DestroySemaphore( VkSemaphore aFence )
    {
        if( mSemaphores.find( aFence ) != mSemaphores.end() )
        {
            mSemaphores.erase( aFence );
            vkDestroySemaphore( mVkLogicalDevice, aFence, nullptr );
        }
    }

    void VkGraphicContext::ResetFences( std::vector<VkFence> aFences )
    {
        std::vector<VkFence> aNonNullFences;
        for( auto &lFence : aFences )
        {
            if( lFence != VK_NULL_HANDLE ) aNonNullFences.push_back( lFence );
        }

        if( aNonNullFences.size() != 0 ) vkResetFences( mVkLogicalDevice, aNonNullFences.size(), aNonNullFences.data() );
    }

    void VkGraphicContext::ResetFence( VkFence aFence ) { ResetFences( { aFence } ); }

    void VkGraphicContext::WaitForFences( std::vector<VkFence> aFences, uint64_t aTimeout )
    {
        std::vector<VkFence> aNonNullFences;
        for( auto &lFence : aFences )
        {
            if( lFence != VK_NULL_HANDLE ) aNonNullFences.push_back( lFence );
        }

        if( aNonNullFences.size() != 0 )
            vkWaitForFences( mVkLogicalDevice, aNonNullFences.size(), aNonNullFences.data(), VK_TRUE, aTimeout );
    }

    void VkGraphicContext::WaitForFences( std::vector<VkFence> aFences )
    {
        WaitForFences( aFences, std::numeric_limits<uint64_t>::max() );
    }
    void VkGraphicContext::WaitForFence( VkFence aFence, uint64_t aTimeout ) { WaitForFences( { aFence }, aTimeout ); }
    void VkGraphicContext::WaitForFence( VkFence aFence ) { WaitForFences( { aFence } ); }

    std::tuple<VkFormat, uint32_t, VkExtent2D, VkSwapchainKHR> VkGraphicContext::CreateSwapChain()
    {
        sSwapChainSupportDetails lSwapChainSupport = QuerySwapChainSupport( mVkPhysicalDevice, mVkSurface );

        VkSurfaceFormatKHR lSurfaceFormat   = ChooseSwapSurfaceFormat( lSwapChainSupport.mFormats );
        VkPresentModeKHR   lPresentMode     = ChooseSwapPresentMode( lSwapChainSupport.mPresentModes );
        VkExtent2D         lSwapchainExtent = ChooseSwapExtent( mWindow->GetExtent(), lSwapChainSupport.mCapabilities );

        VkFormat lSwapChainImageFormat = lSurfaceFormat.format;

        uint32_t lSwapChainImageCount = lSwapChainSupport.mCapabilities.minImageCount + 1;
        if( ( lSwapChainSupport.mCapabilities.maxImageCount > 0 ) &&
            ( lSwapChainImageCount > lSwapChainSupport.mCapabilities.maxImageCount ) )
            lSwapChainImageCount = lSwapChainSupport.mCapabilities.maxImageCount;

        VkSwapchainCreateInfoKHR lSwapChainCreateInfo{};
        lSwapChainCreateInfo.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        lSwapChainCreateInfo.surface          = mVkSurface;
        lSwapChainCreateInfo.minImageCount    = lSwapChainImageCount;
        lSwapChainCreateInfo.imageFormat      = lSurfaceFormat.format;
        lSwapChainCreateInfo.imageColorSpace  = lSurfaceFormat.colorSpace;
        lSwapChainCreateInfo.imageExtent      = lSwapchainExtent;
        lSwapChainCreateInfo.imageArrayLayers = 1;
        lSwapChainCreateInfo.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        uint32_t lQueueFamilyIndices[] = { mGraphicFamily, mPresentFamily };

        if( mGraphicFamily != mPresentFamily )
        {
            lSwapChainCreateInfo.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
            lSwapChainCreateInfo.queueFamilyIndexCount = 2;
            lSwapChainCreateInfo.pQueueFamilyIndices   = lQueueFamilyIndices;
        }
        else
        {
            lSwapChainCreateInfo.imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE;
            lSwapChainCreateInfo.queueFamilyIndexCount = 0;
            lSwapChainCreateInfo.pQueueFamilyIndices   = nullptr;
        }

        lSwapChainCreateInfo.preTransform   = lSwapChainSupport.mCapabilities.currentTransform;
        lSwapChainCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        lSwapChainCreateInfo.presentMode    = lPresentMode;
        lSwapChainCreateInfo.clipped        = VK_TRUE;
        lSwapChainCreateInfo.oldSwapchain   = VK_NULL_HANDLE;

        VkSwapchainKHR lNewSwapchain;
        VK_CHECK_RESULT( vkCreateSwapchainKHR( mVkLogicalDevice, &lSwapChainCreateInfo, nullptr, &lNewSwapchain ) );
        mSwapChains.emplace( lNewSwapchain );

        return { lSwapChainImageFormat, lSwapChainImageCount, lSwapchainExtent, lNewSwapchain };
    }

    void VkGraphicContext::DestroySwapChain( VkSwapchainKHR aSwapchain )
    {
        WaitIdle();

        if( mSwapChains.find( aSwapchain ) != mSwapChains.end() )
        {
            mSwapChains.erase( aSwapchain );
            vkDestroySwapchainKHR( mVkLogicalDevice, aSwapchain, nullptr );
        }
    }

    std::vector<VkImage> VkGraphicContext::GetSwapChainImages( VkSwapchainKHR aSwapChain )
    {
        uint32_t lImageCount;
        VK_CHECK_RESULT( vkGetSwapchainImagesKHR( mVkLogicalDevice, aSwapChain, &lImageCount, nullptr ) );

        std::vector<VkImage> lImages( lImageCount );
        VK_CHECK_RESULT( vkGetSwapchainImagesKHR( mVkLogicalDevice, aSwapChain, &lImageCount, lImages.data() ) );

        return lImages;
    }

    VkShaderModule VkGraphicContext::CreateShaderModule( std::vector<uint32_t> aByteCode )
    {
        VkShaderModuleCreateInfo lShaderModuleCreateInfo{};
        lShaderModuleCreateInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        lShaderModuleCreateInfo.pCode    = aByteCode.data();
        lShaderModuleCreateInfo.codeSize = aByteCode.size() * sizeof( uint32_t );

        VkShaderModule lNewShaderModule{};
        VK_CHECK_RESULT( vkCreateShaderModule( mVkLogicalDevice, &lShaderModuleCreateInfo, nullptr, &lNewShaderModule ) );
        mShaderModules.emplace( lNewShaderModule );

        return lNewShaderModule;
    }

    void VkGraphicContext::DestroyShaderModule( VkShaderModule aShaderModule )
    {
        if( mShaderModules.find( aShaderModule ) != mShaderModules.end() )
        {
            mShaderModules.erase( aShaderModule );
            vkDestroyShaderModule( mVkLogicalDevice, aShaderModule, nullptr );
        }
    }

    VkDescriptorSetLayout VkGraphicContext::CreateDescriptorSetLayout( std::vector<VkDescriptorSetLayoutBinding> aBindings,
                                                                       bool                                      aUnbounded )
    {
        VkDescriptorSetLayoutCreateInfo lDescriptorSetLayoutCreateInfo{};
        lDescriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lDescriptorSetLayoutCreateInfo.bindingCount = aBindings.size();
        lDescriptorSetLayoutCreateInfo.pBindings    = aBindings.data();
        lDescriptorSetLayoutCreateInfo.flags        = 0;
        lDescriptorSetLayoutCreateInfo.pNext        = nullptr;

        std::vector<VkDescriptorBindingFlagsEXT> bindFlag( aBindings.size() );

        VkDescriptorSetLayoutBindingFlagsCreateInfoEXT extendedInfo{};
        if( aUnbounded )
        {
            for( uint32_t i = 0; i < aBindings.size(); i++ ) bindFlag[i] = 0;

            bindFlag[aBindings.size() - 1] =
                VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;

            extendedInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT;
            extendedInfo.bindingCount  = bindFlag.size();
            extendedInfo.pBindingFlags = bindFlag.data();
            extendedInfo.pNext         = nullptr;

            lDescriptorSetLayoutCreateInfo.pNext = &extendedInfo;
        }

        VkDescriptorSetLayout lNewDescriptorSetLayout{};
        VK_CHECK_RESULT(
            vkCreateDescriptorSetLayout( mVkLogicalDevice, &lDescriptorSetLayoutCreateInfo, nullptr, &lNewDescriptorSetLayout ) );
        mDescriptorSetLayouts.emplace( lNewDescriptorSetLayout );

        return lNewDescriptorSetLayout;
    }

    void VkGraphicContext::DestroyDescriptorSetLayout( VkDescriptorSetLayout aDescriptorSetLayout )
    {
        if( mDescriptorSetLayouts.find( aDescriptorSetLayout ) != mDescriptorSetLayouts.end() )
        {
            mDescriptorSetLayouts.erase( aDescriptorSetLayout );
            vkDestroyDescriptorSetLayout( mVkLogicalDevice, aDescriptorSetLayout, nullptr );
        }
    }

    VkDescriptorPool VkGraphicContext::CreateDescriptorPool( uint32_t                          aDescriptorSetCount,
                                                             std::vector<VkDescriptorPoolSize> aPoolSizes )
    {
        VkDescriptorPoolCreateInfo lDescriptorPoolCreateInfo{};
        lDescriptorPoolCreateInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        lDescriptorPoolCreateInfo.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        lDescriptorPoolCreateInfo.maxSets       = aDescriptorSetCount;
        lDescriptorPoolCreateInfo.poolSizeCount = (uint32_t)aPoolSizes.size();
        lDescriptorPoolCreateInfo.pPoolSizes    = aPoolSizes.data();

        VkDescriptorPool lNewDescriptorPool{};
        VK_CHECK_RESULT( vkCreateDescriptorPool( mVkLogicalDevice, &lDescriptorPoolCreateInfo, nullptr, &lNewDescriptorPool ) );
        mDescriptorPools.emplace( lNewDescriptorPool );

        return lNewDescriptorPool;
    }

    VkDescriptorSet VkGraphicContext::AllocateDescriptorSet( VkDescriptorSetLayout aLayout, uint32_t aDescriptorCount )
    {
        VkDescriptorSetAllocateInfo lDescriptorSetAllocInfo{};
        lDescriptorSetAllocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        lDescriptorSetAllocInfo.pNext              = nullptr;
        lDescriptorSetAllocInfo.descriptorPool     = mVkDescriptorPool;
        lDescriptorSetAllocInfo.descriptorSetCount = 1;

        uint32_t lDescriptorCount[1];
        if( aDescriptorCount > 0 )
        {
            lDescriptorCount[0] = aDescriptorCount;

            VkDescriptorSetVariableDescriptorCountAllocateInfo lDescriptorCountInfo{};
            lDescriptorCountInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO;
            lDescriptorCountInfo.descriptorSetCount = 1;
            lDescriptorCountInfo.pDescriptorCounts  = lDescriptorCount;

            lDescriptorSetAllocInfo.pNext = &lDescriptorCountInfo;
        }

        VkDescriptorSetLayout lLayouts[] = { aLayout };

        lDescriptorSetAllocInfo.pSetLayouts = lLayouts;

        VkDescriptorSet lNewDescriptorSet;
        VK_CHECK_RESULT( vkAllocateDescriptorSets( mVkLogicalDevice, &lDescriptorSetAllocInfo, &lNewDescriptorSet ) );

        return lNewDescriptorSet;
    }

    void VkGraphicContext::FreeDescriptorSet( VkDescriptorSet *aDescriptorSet, uint32_t aDescriptorCount )
    {
        VK_CHECK_RESULT( vkFreeDescriptorSets( mVkLogicalDevice, mVkDescriptorPool, 1, aDescriptorSet ) );
    }

    void VkGraphicContext::DestroyDescriptorPool( VkDescriptorPool aDescriptorPool )
    {
        if( mDescriptorPools.find( aDescriptorPool ) != mDescriptorPools.end() )
        {
            mDescriptorPools.erase( aDescriptorPool );
            vkDestroyDescriptorPool( mVkLogicalDevice, aDescriptorPool, nullptr );
        }
    }

    VkPipelineLayout VkGraphicContext::CreatePipelineLayout( std::vector<VkDescriptorSetLayout> aDescriptorSetLayout,
                                                             std::vector<VkPushConstantRange>   aPushConstants )
    {
        VkPipelineLayoutCreateInfo lPipelineLayoutCreateInfo{};
        lPipelineLayoutCreateInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        lPipelineLayoutCreateInfo.pNext                  = nullptr;
        lPipelineLayoutCreateInfo.flags                  = 0;
        lPipelineLayoutCreateInfo.setLayoutCount         = aDescriptorSetLayout.size();
        lPipelineLayoutCreateInfo.pSetLayouts            = aDescriptorSetLayout.data();
        lPipelineLayoutCreateInfo.pPushConstantRanges    = aPushConstants.data();
        lPipelineLayoutCreateInfo.pushConstantRangeCount = aPushConstants.size();

        VkPipelineLayout lNewPipelineLayout{};
        if( vkCreatePipelineLayout( mVkLogicalDevice, &lPipelineLayoutCreateInfo, nullptr, &lNewPipelineLayout ) != VK_SUCCESS )
        {
            return VK_NULL_HANDLE;
        }

        mPipelineLayouts.emplace( lNewPipelineLayout );
        return lNewPipelineLayout;
    }

    void VkGraphicContext::DestroyPipelineLayout( VkPipelineLayout aPipelineLayout )
    {
        if( mPipelineLayouts.find( aPipelineLayout ) != mPipelineLayouts.end() )
        {
            mPipelineLayouts.erase( aPipelineLayout );
            vkDestroyPipelineLayout( mVkLogicalDevice, aPipelineLayout, nullptr );
        }
    }

    VkResult VkGraphicContext::AcquireNextImage( VkSwapchainKHR aSwapChain, uint64_t aTimeout, VkSemaphore aWaitSemaphore,
                                                 uint32_t *aNewImageIndex )
    {
        return vkAcquireNextImageKHR( mVkLogicalDevice, aSwapChain, aTimeout, aWaitSemaphore, VK_NULL_HANDLE, aNewImageIndex );
    }

    VkResult VkGraphicContext::Present( VkSwapchainKHR aSwapChain, uint32_t aImageIndex, VkSemaphore aWaitSemaphore )
    {
        VkPresentInfoKHR lPresentInfo{};
        lPresentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        VkSemaphore lWaitSemaphores[]   = { aWaitSemaphore };
        lPresentInfo.waitSemaphoreCount = 1;
        lPresentInfo.pWaitSemaphores    = lWaitSemaphores;

        VkSwapchainKHR lSwapChains[] = { aSwapChain };
        uint32_t       lImageIndex   = aImageIndex;
        lPresentInfo.swapchainCount  = 1;
        lPresentInfo.pSwapchains     = lSwapChains;
        lPresentInfo.pImageIndices   = &lImageIndex;

        return vkQueuePresentKHR( mVkPresentQueue, &lPresentInfo );
    }

    void VkGraphicContext::UpdateDescriptorSets( VkWriteDescriptorSet aWriteOps )
    {
        vkUpdateDescriptorSets( mVkLogicalDevice, 1, &aWriteOps, 0, nullptr );
    }

    void VkGraphicContext::WaitIdle() { vkDeviceWaitIdle( mVkLogicalDevice ); }
    void VkGraphicContext::WaitIdle( VkQueue aQueue ) { vkQueueWaitIdle( aQueue ); };

} // namespace SE::Graphics