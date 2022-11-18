#pragma once

#include "Core/Math/Types.h"

#include <memory>
#include <vulkan/vulkan.h>

#include "VkContext.h"
#include "VkImage.h"

#include "VkAbstractRenderPass.h"

#include "Core/Memory.h"

namespace SE::Graphics::Internal
{
    using namespace SE::Core;

    struct sVkRenderPassObject : public sVkAbstractRenderPassObject
    {
        sVkRenderPassObject()                        = default;
        sVkRenderPassObject( sVkRenderPassObject & ) = default;
        sVkRenderPassObject( Ref<VkContext> aContext, std::vector<VkAttachmentDescription> aAttachments,
                             std::vector<VkSubpassDescription> aSubpasses, std::vector<VkSubpassDependency> aSubpassDependencies );

        sVkRenderPassObject( Ref<VkContext> aContext, VkFormat aFormat, uint32_t aSampleCount, bool aIsSampled, bool aIsPresented,
                             math::vec4 aClearColor );

        ~sVkRenderPassObject() = default;
    };

} // namespace SE::Graphics::Internal
