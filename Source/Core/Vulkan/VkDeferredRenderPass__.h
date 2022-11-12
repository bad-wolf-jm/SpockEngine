#pragma once

#include "Core/Math/Types.h"
#include "Core/Platform/ViewportClient.h"
#include <memory>
#include <vulkan/vulkan.h>

#include "VkContext.h"
#include "VkImage.h"
#include "VkAbstractRenderPass.h"

#include "Core/Memory.h"

namespace LTSE::Graphics::Internal
{
    using namespace LTSE::Core;

    struct sVkDeferredRenderPassObject : public sVkAbstractRenderPassObject
    {
        sVkDeferredRenderPassObject()                        = default;
        sVkDeferredRenderPassObject( sVkDeferredRenderPassObject & ) = default;
        sVkDeferredRenderPassObject( Ref<VkContext> aContext, std::vector<VkAttachmentDescription> aAttachments,
            std::vector<VkSubpassDescription> aSubpasses, std::vector<VkSubpassDependency> aSubpassDependencies );

        sVkDeferredRenderPassObject( Ref<VkContext> aContext, VkFormat aFormat, uint32_t aSampleCount, bool aIsSampled, bool aIsPresented,
            math::vec4 aClearColor );

        ~sVkDeferredRenderPassObject() = default;
    };

} // namespace LTSE::Graphics::Internal
