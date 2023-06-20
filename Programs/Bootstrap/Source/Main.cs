using System;
using System.IO;
using System.Linq;
using System.Drawing;
using System.Drawing.Text;
using System.Windows.Media;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using CommandLine;
using SpockEngine;
using Math = SpockEngine.Math;

using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

class SEBootstrap
{
    public class Options
    {
        [Option('a', "application", Required = true, HelpText = "Run application")]
        public string Application { get; set; }

        [Option('x', Required = false, HelpText = "Window x position")]
        public int X { get; set; }

        [Option('y', Required = false, HelpText = "Window y position")]
        public int Y { get; set; }

        [Option('w', "width", Required = false, HelpText = "Window width")]
        public int Width { get; set; }

        [Option('h', "height", Required = false, HelpText = "Window height")]
        public int Height { get; set; }
    }

    static string CreateFolder(string[] aPathElements)
    {
        var lPath = Path.Combine(aPathElements);
        if (!Directory.Exists(lPath))
            Directory.CreateDirectory(lPath);

        return lPath;
    }


    static void UpdateDelegate(float aTs)
    {

    }

    static bool RenderUIDelegate(float aTs)
    {
        return false;
    }

    static void RenderDelegate()
    {

    }

    delegate void UpdateDelegateType(float aTs);
    delegate void RenderDelegateType();
    delegate bool RenderUIDelegateType(float aTs);

    static void GuardedMain(Options aOpt)
    {
        var lExePath = Directory.GetParent(System.Reflection.Assembly.GetExecutingAssembly().Location).FullName;
        var lUserHome = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        var lUserConfig = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);

        var lUIConfiguration = new UIConfiguration();
        lUIConfiguration.mFontSize = 15.0f;
        lUIConfiguration.mMainFont = Path.Combine(new string[] { lExePath, "Fonts", "Roboto-Thin.ttf" });
        lUIConfiguration.mBoldFont = Path.Combine(new string[] { lExePath, "Fonts", "Roboto-Bold.ttf" });
        lUIConfiguration.mItalicFont = Path.Combine(new string[] { lExePath, "Fonts", "Roboto-Italic.ttf" });
        lUIConfiguration.mBoldItalicFont = Path.Combine(new string[] { lExePath, "Fonts", "Roboto-BoldItalic.ttf" });
        lUIConfiguration.mIconFont = Path.Combine(new string[] { lExePath, "Fonts", "fontawesome-webfont.ttf" });
        lUIConfiguration.mMonoFont = Path.Combine(new string[] { lExePath, "Fonts", "DejaVuSansMono.ttf" });

        var lAppConfigRoot = Path.Combine(lUserConfig, aOpt.Application);

        var lAppConfig = CreateFolder(new string[] { lAppConfigRoot, "Config" });
        var lAppLogs = CreateFolder(new string[] { lAppConfigRoot, "Logs" });

        lUIConfiguration.mIniFile = Path.Combine(lAppConfig, "imgui.ini");
        var lAppIniConfig = Path.Combine(lAppConfig, "Application.yaml");

        float lX = 0.0f;
        float lY = 0.0f;
        float lWidth = 1920.0f;
        float lHeight = 1000.0f;

        UpdateDelegateType lUpdateDelegatePtr = UpdateDelegate;
        RenderDelegateType lRenderDelegateTypePtr = RenderDelegate;
        RenderUIDelegateType lRenderUIDelegateTypePtr = RenderUIDelegate;

        CppCall.Engine_Initialize(new Math.vec2(lX, lY), new Math.vec2(lWidth, lHeight), lUIConfiguration);
        CppCall.Engine_Main(
            Marshal.GetFunctionPointerForDelegate(lUpdateDelegatePtr),
            Marshal.GetFunctionPointerForDelegate(lRenderDelegateTypePtr),
            Marshal.GetFunctionPointerForDelegate(lRenderUIDelegateTypePtr));
        CppCall.Engine_Shutdown();
    }

    static void Main(string[] args)
    {
        Parser.Default.ParseArguments<Options>(args).WithParsed<Options>(GuardedMain);
    }
}