using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SpockEngine
{
    public class UIMenuItem : UIComponent
    {
        private bool mDerived = false;
        public UIMenuItem() : this(Interop.UIMenuItem_Create()) { }

        public UIMenuItem(IntPtr aDerived) : base(aDerived) { mDerived = false; }

        public UIMenuItem(IntPtr aDerived, bool aIsDerived) : base(aDerived) { mDerived = aIsDerived; }

        public UIMenuItem(string aText) : this()
        {
            SetText(aText);
        }

        public UIMenuItem(string aText, string aShortcut) : this()
        {
            SetText(aText);
            SetShortcut(aShortcut);
        }

        ~UIMenuItem() { if (!mDerived) Interop.UIMenuItem_Destroy(mInstance); }

        public void SetText(string aText)
        {
            Interop.UIMenuItem_SetText(mInstance, aText);
        }

        public void SetShortcut(string aShortcut)
        {
            Interop.UIMenuItem_SetShortcut(mInstance, aShortcut);
        }

        public void SetTextColor(Math.vec4 aColor)
        {
            Interop.UIMenuItem_SetTextColor(mInstance, aColor);
        }

        public delegate void TriggeredDelegate();
        TriggeredDelegate onTriggered;
        public void OnTrigger(TriggeredDelegate aHandler)
        {
            onTriggered = aHandler;

            Interop.UIMenuItem_OnTrigger(mInstance, Marshal.GetFunctionPointerForDelegate(onTriggered));
        }
    }

    public class UIMenuSeparator : UIMenuItem
    {
        public UIMenuSeparator() : this(Interop.UIMenuSeparator_Create()) { }

        public UIMenuSeparator(IntPtr aDerived) : base(aDerived, true) { }

        ~UIMenuSeparator() { Interop.UIMenuSeparator_Destroy(mInstance); }
    }



    public class UIMenu : UIMenuItem
    {
        private List<UIMenuItem> mMenuItems;

        public UIMenu() : base(Interop.UIMenu_Create(), true) { mMenuItems = new List<UIMenuItem>(); }

        public UIMenu(IntPtr aDerived) : base(aDerived, true) { mMenuItems = new List<UIMenuItem>(); }

        public UIMenu(string aText) : this(Interop.UIMenu_CreateWithText(aText)) { mMenuItems = new List<UIMenuItem>(); }

        ~UIMenu() { Interop.UIMenu_Destroy(mInstance); }

        public UIMenu AddMenu(string aName)
        {
            var lNewMenu = new UIMenu(Interop.UIMenu_AddMenu(mInstance, aName));
            mMenuItems.Add(lNewMenu);

            return lNewMenu;
        }

        public UIMenuSeparator AddSeparator()
        {
            var lNewMenu = new UIMenuSeparator(Interop.UIMenu_AddSeparator(mInstance));
            mMenuItems.Add(lNewMenu);

            return lNewMenu;
        }

        public UIMenuItem AddAction(string aName, string aShortcut)
        {
            var lNewMenu = new UIMenuItem(Interop.UIMenu_AddAction(mInstance, aName, aShortcut));
            mMenuItems.Add(lNewMenu);

            return lNewMenu;
        }

        public void Update() { Interop.UIMenu_Update(mInstance); }
    }



}