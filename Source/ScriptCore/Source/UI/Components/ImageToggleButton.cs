using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIImageToggleButton : UIBaseImage
    {
        public UIImageToggleButton() : base(UIImageToggleButton_Create()) { }

        ~UIImageToggleButton() { UIImageToggleButton_Destroy(mInstance); }

        public void SetActiveImage(UIBaseImage aImage)
        {
            UIImageToggleButton_SetActiveImage(mInstance, aImage.Instance);
        }

        public void SetInactiveImage(UIBaseImage aImage)
        {
            UIImageToggleButton_SetInactiveImage(mInstance, aImage.Instance);
        }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIImageToggleButton_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIImageToggleButton_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIImageToggleButton_SetActiveImage(ulong aInstance, ulong aImage);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIImageToggleButton_SetInactiveImage(ulong aInstance, ulong aImage);
    }
}
