#pragma once

#include <string>

#include <Core/Math/Types.h>

class PopupWindow
{
  public:
    bool        Visible = false;
    std::string Title   = "";
    math::vec2  Size    = { 0.0f, 0.0f };

  public:
    PopupWindow() = default;
    PopupWindow( std::string a_Title, math::vec2 a_Size );
    ~PopupWindow() = default;

    virtual void WindowContent() = 0;
    void         Display();

  private:
};