/// @file   Timestep.h
///
/// @brief  Abstract type for a timestep
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert. All rights reserved.

#pragma once

class Timestep
{
  public:
    Timestep( float aTime = 0.0f )
        : mTime( aTime )
    {
    }

    operator float() const { return mTime; }
    float GetMilliseconds() const { return mTime; }
    float GetSeconds() const { return mTime / 1000.0f; }

  private:
    float mTime = 0.0f;
};
