using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIMenuItem : UIComponent
    {
        public UIMenuItem() : base(UIMenuItem_Create()) { }

        public UIMenuItem(ulong aDerived) : base(aDerived) { }

        public UIMenuItem(string aText) : this(UIMenuItem_CreateWithText(aText)) { }

        public UIMenuItem(string aText, string aShortcut) : this(UIMenuItem_CreateWithTextAndShortcut(aText, aShortcut)) { }

        ~UIMenuItem() { UIMenuItem_Destroy(mInstance); }

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

            UIMenuItem_OnTrigger(mInstance, onTriggered);
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenuItem_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenuItem_CreateWithTextAndShortcut(string aText, string aShortcut);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenuItem_CreateWithText(string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenuItem_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenuItem_SetText(ulong aInstance, string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenuItem_SetShortcut(ulong aInstance, string aShortcut);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenuItem_SetTextColor(ulong aInstance, Math.vec4 aColor);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenuItem_OnTrigger(ulong aInstance, TriggeredDelegate aHandler);
    }

    public class UIMenuSeparator : UIMenuItem
    {
        public UIMenuSeparator() : this(UIMenuSeparator_Create()) { }

        public UIMenuSeparator(ulong aDerived) : base(aDerived) { }

        ~UIMenuSeparator() { UIMenuSeparator_Destroy(mInstance); }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenuSeparator_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenuSeparator_Destroy(ulong aInstance);
    }



    public class UIMenu : UIMenuItem
    {
        public UIMenu() : base(UIMenu_Create()) { }

        public UIMenu(ulong aDerived) : base(aDerived) { }

        public UIMenu(string aText) : this(UIMenu_CreateWithText(aText)) { }

        ~UIMenu() { UIMenu_Destroy(mInstance); }

        public UIMenu AddMenu(string aName)
        {
            return new UIMenu(UIMenu_AddMenu(mInstance, aName));
        }

        public UIMenuSeparator AddSeparator(string aName)
        {
            return new UIMenuSeparator(UIMenu_AddSeparator(mInstance));
        }

        public UIMenuItem AddAction(string aName, string aShortcut)
        {
            return new UIMenuSeparator(UIMenu_AddAction(mInstance, aName, aShortcut));
        }


        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenu_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenu_CreateWithText(string aText);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenu_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenu_AddAction(ulong aInstance, string aName, string aShortcut);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenu_AddSeparator(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIMenu_AddMenu(ulong aInstance, string aName);

    }



}