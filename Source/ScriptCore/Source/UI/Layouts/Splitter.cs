using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public enum eBoxLayoutOrientation
    {
        HORIZONTAL,
        VERTICAL
    }

    public class UISplitter : UIComponent
    {
        private bool mDerived = false;
        public UISplitter() : this(UISplitter_Create(), false) { }

        public UISplitter(ulong aInstance, bool aDerived) : base(aInstance) { mDerived = aDerived; }

        public UISplitter(eBoxLayoutOrientation aOrientation) : this(UISplitter_CreateWithOrientation(aOrientation), false) { }

        ~UISplitter() { if (!mDerived) UISplitter_Destroy(mInstance); }

        public void SetItemSpacing(float aItemSpacing)
        {
            UISplitter_SetItemSpacing(mInstance, aItemSpacing);
        }

        public void Add1(UIComponent aChild)
        {
            var lInstance = (aChild != null) ? aChild.Instance : 0;

            UISplitter_Add1(mInstance, lInstance);
        }

        public void Add2(UIComponent aChild)
        {
            var lInstance = (aChild != null) ? aChild.Instance : 0;

            UISplitter_Add2(mInstance, lInstance);
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UISplitter_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UISplitter_CreateWithOrientation(eBoxLayoutOrientation aOrientation);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UISplitter_Add1(ulong aInstance, ulong aChild);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UISplitter_Add2(ulong aInstance, ulong aChild);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UISplitter_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UISplitter_SetItemSpacing(ulong aInstance, float aItemSpacing);

    }
}