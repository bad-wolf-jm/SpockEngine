using System;
using System.Runtime.CompilerServices;

namespace SpockEngine
{
    public class UIForm : UIComponent
    {
        public UIForm() : base(UIForm_Create()) { }

        ~UIForm() { UIForm_Destroy(mInstance); }

        public void SetTitle(string aTitle) { UIForm_SetTitle(mInstance, aTitle); }

        public void SetContent(UIComponent aContent) { UIForm_SetContent(mInstance, aContent.Instance); }

        public override void Update() { UIForm_Update(mInstance); }

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static ulong UIForm_Create();

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIForm_Destroy(ulong aInstance);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIForm_SetTitle(ulong aInstance, string aTitle);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIForm_SetContent(ulong aInstance, ulong aContent);

        [MethodImplAttribute(MethodImplOptions.InternalCall)]
        private extern static void UIForm_Update(ulong aInstance);
    }
}
