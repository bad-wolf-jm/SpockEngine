using System.Runtime.InteropServices;
using SpockEngine;

namespace SpockEngine
{
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct UIConfiguration
    {
        [MarshalAs(UnmanagedType.LPWStr)]
        public string mIniFile;

        public float mFontSize;

        [MarshalAs(UnmanagedType.LPWStr)]
        public string mMainFont;

        [MarshalAs(UnmanagedType.LPWStr)]
        public string mBoldFont;

        [MarshalAs(UnmanagedType.LPWStr)]
        public string mItalicFont;

        [MarshalAs(UnmanagedType.LPWStr)]
        public string mBoldItalicFont;

        [MarshalAs(UnmanagedType.LPWStr)]
        public string mMonoFont;

        [MarshalAs(UnmanagedType.LPWStr)]
        public string mIconFont;
    }


    public class SEApplication
    {
        public SEApplication() { }

        public virtual void Initialize(string aConfigurationPath) { }

        public virtual void Shutdown(string aConfigurationPath) { }

        public virtual void Update(float aTs) { }

        public virtual bool UpdateMenu() { return false; }

        public virtual void UpdateUI(float aTs) { }
    }
}
