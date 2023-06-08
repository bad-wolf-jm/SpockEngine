using System;
using System.Runtime.CompilerServices;

using SpockEngine.Math;

namespace SpockEngine
{
    public class UIZLayout : UIComponent
    {
        public UIZLayout() : base(Interop.UIZLayout_Create()) { }

        ~UIZLayout() { Interop.UIZLayout_Destroy(mInstance); }

        public void Add(UIComponent aChild, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment)
        {
            Interop.UIZLayout_AddAlignedNonFixed(mInstance, aChild.Instance, aExpand, aFill, aHAlignment, aVAlignment);
        }

        public void Add(UIComponent aChild, bool aExpand, bool aFill)
        {
            Interop.UIZLayout_AddNonAlignedNonFixed(mInstance, aChild.Instance, aExpand, aFill);
        }

        public void Add(UIComponent aChild, SpockEngine.Math.vec2 aSize, SpockEngine.Math.vec2 aPosition, bool aExpand, bool aFill)
        {
            Interop.UIZLayout_AddNonAlignedFixed(mInstance, aChild.Instance, aSize, aPosition, aExpand, aFill);
        }

        public void Add(UIComponent aChild, SpockEngine.Math.vec2 aSize, SpockEngine.Math.vec2 aPosition, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment)
        {
            Interop.UIZLayout_AddAlignedFixed(mInstance, aChild.Instance, aSize, aPosition, aExpand, aFill, aHAlignment, aVAlignment);
        }
    }
}