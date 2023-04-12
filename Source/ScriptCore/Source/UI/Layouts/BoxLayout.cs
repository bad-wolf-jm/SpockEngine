using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public enum eBoxLayoutOrientation
    {
        HORIZONTAL,
        VERTICAL
    }

    public class UIBoxLayout : UIComponent
    {
        private bool mDerived = false;
        public UIBoxLayout() { }

        public UIBoxLayout(ulong aInstance, bool aDerived) : base(aInstance) { mDerived = aDerived; }

        public UIBoxLayout(eBoxLayoutOrientation aOrientation) : this(UIBoxLayout_CreateWithOrientation(aOrientation), false) { }

        ~UIBoxLayout() { if (!mDerived) UIBoxLayout_Destroy(mInstance); }

        public void SetItemSpacing(float aItemSpacing)
        {
            UIBoxLayout_SetItemSpacing(mInstance, aItemSpacing);
        }

        public void Add(UIComponent aChild, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment)
        {
            var lInstance = (aChild != null) ? aChild.Instance : 0;
            UIBoxLayout_AddAlignedNonFixed(mInstance, lInstance, aExpand, aFill, aHAlignment, aVAlignment);
        }

        public void Add(UIComponent aChild, bool aExpand, bool aFill)
        {
            var lInstance = (aChild != null) ? aChild.Instance : 0;
            UIBoxLayout_AddNonAlignedNonFixed(mInstance, lInstance, aExpand, aFill);
        }

        public void Add(UIComponent aChild, float aFixedSize, bool aExpand, bool aFill)
        {
            var lInstance = (aChild != null) ? aChild.Instance : 0;
            UIBoxLayout_AddNonAlignedFixed(mInstance, lInstance, aFixedSize, aExpand, aFill);
        }

        public void Add(UIComponent aChild, float aFixedSize, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment)
        {
            var lInstance = (aChild != null) ? aChild.Instance : 0;
            UIBoxLayout_AddAlignedFixed(mInstance, lInstance, aFixedSize, aExpand, aFill, aHAlignment, aVAlignment);
        }

        public void Clear()
        {
            UIBoxLayout_Clear(mInstance);
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIBoxLayout_CreateWithOrientation(eBoxLayoutOrientation aOrientation);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBoxLayout_AddAlignedNonFixed(ulong aInstance, ulong aChild, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBoxLayout_AddNonAlignedNonFixed(ulong aInstance, ulong aChild, bool aExpand, bool aFill);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBoxLayout_AddAlignedFixed(ulong aInstance, ulong aChild, float aFixedSize, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBoxLayout_AddNonAlignedFixed(ulong aInstance, ulong aChild, float aFixedSize, bool aExpand, bool aFill);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBoxLayout_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBoxLayout_SetItemSpacing(ulong aInstance, float aItemSpacing);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIBoxLayout_Clear(ulong aInstance);

    }
}