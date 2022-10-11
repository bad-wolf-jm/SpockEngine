/// @file   SensorComponents.h
///
/// @brief  Definitions for sensor component types
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2022 LeddarTech Inc. All rights reserved.

#pragma once

#include <filesystem>
#include <variant>
#include <vector>

#include "FileIO.h"

namespace LTSE::SensorModel
{
    namespace fs = std::filesystem;

    /// @brief Sampler
    ///
    /// This component roughly corresponds to an analog-digital converter. The laser pulse returned from the target is sampled
    /// at the given frequency. Here the length member specifies the number of samples to collect in order to produce a waveform.
    ///
    /// ```yaml
    /// type: sampler
    /// name: sampler_1
    /// data:
    ///     length: 1024
    ///     frequency: 150000000.0
    /// ```
    ///
    struct sSamplerComponentData
    {
        uint32_t mLength = 1024;         //!< number of samples to gather for a given acquisition
        float mFrequency = 800000000.0f; //!< Sampling frequency

        /// @brief Default constructor
        sSamplerComponentData() = default;

        /// @brief Copy constructor
        sSamplerComponentData( const sSamplerComponentData & ) = default;

        /// @brief Default destructor
        ~sSamplerComponentData() = default;
    };

    /// @brief Supported component types
    ///
    /// To define a new component type, one should extend this enumeration with a new index. The ordering of the
    /// supported component typesa should match the ordering defined in the subsequent variant definition
    ///
    enum class eComponentType : uint32_t
    {
        UNKNOWN = 0,
        SAMPLER = 1
    };

    /// @brief Main component value
    ///
    /// To define a new component type, the structure corresponding to the component should be added at the
    /// end of the variant definition. Node that the ordering of the types in the variant definition should
    /// match the ordering in the enumeration above
    ///
    using SensorComponentValue = std::variant<std::monostate, sSamplerComponentData>;

    /// @brief Abstract sensor component
    ///
    /// Holds the asset data that has been parsed from disk.
    ///
    struct sSensorComponentData
    {
        std::string mName = "";        //!< Name of the component for display purposes
        SensorComponentValue mValue{}; //!< Parsed component data.

        /// @brief Default constructor
        sSensorComponentData() = default;

        /// @brief Copy constructor
        sSensorComponentData( const sSensorComponentData & ) = default;

        /// @brief Default destructor
        ~sSensorComponentData() = default;

        /// @brief Retrieve the structure stored at the type index corresponding to _Ty
        ///
        /// @tparam _Ty Type to retrieve.
        ///
        /// @return The relevant structure.
        template <typename _Ty> _Ty Get() { return std::get<_Ty>( mValue ); }

        /// @brief Retrieve the structure stored at the type index corresponding to _Ty (const version)
        ///
        /// @tparam _Ty Type to retrieve.
        ///
        /// @return The relevant structure.
        template <typename _Ty> _Ty Get() const { return std::get<_Ty>( mValue ); }

        /// @brief Retrieve the type of the currently stored component
        ///
        /// @return The type of the currently stored component.
        ///
        eComponentType Type() { return static_cast<eComponentType>( mValue.index() ); }

        /// @brief Retrieve the type of the currently stored component (const version)
        ///
        /// @return The type of the currently stored component.
        ///
        eComponentType Type() const { return static_cast<eComponentType>( mValue.index() ); }
    };

    /// @brief read a component from a parsed configuration node
    ///
    /// The component is  parsed into the appropriate `sSensorAssetData` structure.
    ///
    /// @param aAssetData Specification data for the component
    ///
    /// @return The parsed component
    ///
    sSensorComponentData ReadComponent( ConfigurationNode const &aAssetData );

} // namespace LTSE::SensorModel
