/// @file   Timestep.h
///
/// @brief  Abstract type for a timestep
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert All rights reserved.

#pragma once

class Timestep
{
  public:
    Timestep( float time = 0.0f )
        : _time( time )
    {
    }

    operator float() const
    {
        return _time;
    }

    float GetMilliseconds() const
    {
        return _time;
    }
    
    float GetSeconds() const
    {
        return _time / 1000.0f;
    }

  private:
    float _time = 0.0f;
};
