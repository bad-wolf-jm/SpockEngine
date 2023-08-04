#pragma once

#include "Core/Math/Types.h"
#include "Graphics/Interface/IRenderPass.h"
#include "Graphics/Interface/IWindow.h"
#include <memory>
#include <vulkan/vulkan.h>

#include "Graphics/Vulkan/VkGraphicContext.h"

#include "Core/Memory.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    struct VkRenderPassObject : public IRenderPass
    {
        VkRenderPass mVkObject    = VK_NULL_HANDLE;
        uint32_t     mSampleCount = 1;

        VkRenderPassObject()                       = default;
        VkRenderPassObject( VkRenderPassObject & ) = default;
        VkRenderPassObject( Ref<VkGraphicContext> aContext, vector_t<VkAttachmentDescription> aAttachments,
                            vector_t<VkSubpassDescription> aSubpasses, vector_t<VkSubpassDependency> aSubpassDependencies );

        VkRenderPassObject( Ref<VkGraphicContext> aContext, VkFormat aFormat, uint32_t aSampleCount, bool aIsSampled,
                            bool aIsPresented, math::vec4 aClearColor );

        ~VkRenderPassObject();

        VkAttachmentDescription ColorAttachment( VkFormat aFormat, uint32_t aSampleCount, bool aIsSampled, bool aIsPresented,
                                                 bool aIsDefined, VkAttachmentLoadOp aAttachmentLoadOp,
                                                 VkAttachmentStoreOp aAttachmentStoreOp );
        VkAttachmentDescription DepthAttachment( bool aIsDefined, uint32_t aSampleCount, VkAttachmentLoadOp aAttachmentLoadOp,
                                                 VkAttachmentStoreOp aAttachmentStoreOp );

        vector_t<VkClearValue> GetClearValues();

        vector_t<VkSubpassDependency> DefaultSubpassDependencies();

        void CreateUnderlyingRenderpass( vector_t<VkAttachmentDescription> aAttachments,
                                         vector_t<VkAttachmentReference>   aAttachmentReferences,
                                         VkAttachmentReference            *aDepthAttachmentReference,
                                         VkAttachmentReference            *aResolveAttachmentReference );

        uint32_t GetColorAttachmentCount()
        {
            return mColorAttachmentCount;
        }

      protected:
        Ref<VkGraphicContext>  mContext              = nullptr;
        vector_t<VkClearValue> mClearValues          = {};
        uint32_t               mColorAttachmentCount = 0;
    };

} // namespace SE::Graphics
