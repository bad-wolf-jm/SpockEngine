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
            UIBoxLayout_AddAlignedNonFixed(mInstance, aChild.Instance, aExpand, aFill, aHAlignment, aVAlignment);
        }

        public void Add(UIComponent aChild, bool aExpand, bool aFill)
        {
            UIBoxLayout_AddNonAlignedNonFixed(mInstance, aChild.Instance, aExpand, aFill);
        }

        public void Add(UIComponent aChild, float aFixedSize, bool aExpand, bool aFill)
        {
            UIBoxLayout_AddNonAlignedFixed(mInstance, aChild.Instance, aFixedSize, aExpand, aFill);
        }

        public void Add(UIComponent aChild, float aFixedSize, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment)
        {
            UIBoxLayout_AddAlignedFixed(mInstance, aChild.Instance, aFixedSize, aExpand, aFill, aHAlignment, aVAlignment);
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

    }
}