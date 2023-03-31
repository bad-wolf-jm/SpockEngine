using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIMenuItem : UIComponent
    {
        private bool mDerived = false;
        public UIMenuItem() : this(UIMenuItem_Create()) { }

        public UIMenuItem(ulong aDerived) : base(aDerived) { mDerived = false; }

        public UIMenuItem(ulong aDerived, bool aIsDerived) : base(aDerived) { mDerived = aIsDerived; }

        public UIMenuItem(string aText) : this(UIMenuItem_CreateWithText(aText)) { }

        public UIMenuItem(string aText, string aShortcut) : this(UIMenuItem_CreateWithTextAndShortcut(aText, aShortcut)) { }

        ~UIMenuItem() { if (!mDerived) UIMenuItem_Destroy(mInstance); }

        public void SetText(string aText)
        {
            UIMenuItem_SetText(mInstance, aText);
        }

        public void SetShortcut(string aShortcut)
        {
            UIMenuItem_SetShortcut(mInstance, aShortcut);
        }

        public void SetTextColor(Math.vec4 aColor)
        {
            UIMenuItem_SetTextColor(mInstance, aColor);
        }

        public delegate void TriggeredDelegate();
        TriggeredDelegate onTriggered;
        public void OnTrigger(TriggeredDelegate aHandler)
        {
            onTriggered = aHandler;

            Console.WriteLine("555555555555555555555");

            UIMenuItem_OnTrigger(mInstance, onTriggered);
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenuItem_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenuItem_CreateWithTextAndShortcut(string aText, string aShortcut);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenuItem_CreateWithText(string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIMenuItem_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIMenuItem_SetText(ulong aInstance, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIMenuItem_SetShortcut(ulong aInstance, string aShortcut);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIMenuItem_SetTextColor(ulong aInstance, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIMenuItem_OnTrigger(ulong aInstance, TriggeredDelegate aHandler);
    }

    public class UIMenuSeparator : UIMenuItem
    {
        public UIMenuSeparator() : this(UIMenuSeparator_Create()) { }

        public UIMenuSeparator(ulong aDerived) : base(aDerived, true) { }

        ~UIMenuSeparator() { UIMenuSeparator_Destroy(mInstance); }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenuSeparator_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIMenuSeparator_Destroy(ulong aInstance);
    }



    public class UIMenu : UIMenuItem
    {
        private List<UIMenuItem> mMenuItems;

        public UIMenu() : base(UIMenu_Create(), true) { mMenuItems = new List<UIMenuItem>(); }

        public UIMenu(ulong aDerived) : base(aDerived, true) { mMenuItems = new List<UIMenuItem>(); }

        public UIMenu(string aText) : this(UIMenu_CreateWithText(aText)) { mMenuItems = new List<UIMenuItem>(); }

        ~UIMenu() { UIMenu_Destroy(mInstance); }

        public UIMenu AddMenu(string aName)
        {
            var lNewMenu = new UIMenu(UIMenu_AddMenu(mInstance, aName));
            mMenuItems.Add(lNewMenu);

            return lNewMenu;
        }

        public UIMenuSeparator AddSeparator()
        {
            var lNewMenu = new UIMenuSeparator(UIMenu_AddSeparator(mInstance));
            mMenuItems.Add(lNewMenu);

            return lNewMenu;
        }

        public UIMenuItem AddAction(string aName, string aShortcut)
        {
            var lNewMenu = new UIMenuItem(UIMenu_AddAction(mInstance, aName, aShortcut));
            mMenuItems.Add(lNewMenu);

            return lNewMenu;
        }

        public void Update() { UIMenu_Update(mInstance); }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenu_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenu_CreateWithText(string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIMenu_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenu_AddAction(ulong aInstance, string aName, string aShortcut);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenu_AddSeparator(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenu_AddMenu(ulong aInstance, string aName);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIMenu_Update(ulong aInstance);
    }



}