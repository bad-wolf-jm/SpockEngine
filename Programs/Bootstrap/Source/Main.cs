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

    private static Dictionary<(System.Windows.FontStyle, System.Windows.FontWeight), string> GetFilesForFont(string fontName)
    {
        var fontNameToFiles = new Dictionary<string, Dictionary<(System.Windows.FontStyle, System.Windows.FontWeight), string>>();

        foreach (var fontFile in Directory.GetFiles(Environment.GetFolderPath(Environment.SpecialFolder.Fonts)))
        {
            var fc = new PrivateFontCollection();

            if (File.Exists(fontFile))
                fc.AddFontFile(fontFile);

            if ((!fc.Families.Any()))
                continue;

            var name = fc.Families[0].Name;

            var lTf = new GlyphTypeface(new Uri(fontFile));
            var lFontStyle = lTf.Style;

            if (!fontNameToFiles.TryGetValue(name, out var files))
            {
                files = new Dictionary<(System.Windows.FontStyle, System.Windows.FontWeight), string>();
                fontNameToFiles[name] = files;
            }

            files[(lFontStyle, lTf.Weight)] = fontFile;
        }

        if (!fontNameToFiles.TryGetValue(fontName, out var result))
            return null;

        return result;
    }

    static string CreateFolder(string[] aPathElements)
    {
        var lPath = Path.Combine(aPathElements);
        if (!Directory.Exists(lPath))
            Directory.CreateDirectory(lPath);

        return lPath;
    }

    static void GuardedMain(Options aOpt)
    {
        var lExePath = Directory.GetParent(System.Reflection.Assembly.GetExecutingAssembly().Location).FullName;
        var lUserHome = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        var lUserConfig = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);

        var lUIConfiguration = new UIConfiguration();
        lUIConfiguration.mFontSize = 15.0f;
        lUIConfiguration.mMainFont = Path.Combine(new string[] {lExePath, "Fonts", "Roboto-Thin.ttf"});
        lUIConfiguration.mBoldFont = Path.Combine(new string[] {lExePath, "Fonts", "Roboto-Bold.ttf"});
        lUIConfiguration.mItalicFont = Path.Combine(new string[] {lExePath, "Fonts", "Roboto-Italic.ttf"});
        lUIConfiguration.mBoldItalicFont = Path.Combine(new string[] {lExePath, "Fonts", "Roboto-BoldItalic.ttf"});
        lUIConfiguration.mIconFont = Path.Combine(new string[] {lExePath, "Fonts", "fontawesome-webfont.ttf"});
        lUIConfiguration.mMonoFont = Path.Combine(new string[] {lExePath, "Fonts", "DejaVuSansMono.ttf"});

        var lAppConfigRoot = Path.Combine(lUserConfig, aOpt.Application);

        var lAppConfig = CreateFolder(new string[] { lAppConfigRoot, "Config" });
        var lAppLogs = CreateFolder(new string[] { lAppConfigRoot, "Logs" });

        lUIConfiguration.mIniFile = Path.Combine(lAppConfig, "imgui.ini");
        var lAppIniConfig = Path.Combine(lAppConfig, "Application.yaml");

        float lX = 0.0f;
        float lY = 0.0f;
        float lWidth = 1920.0f;
        float lHeight = 1000.0f;

        // CppCall.Engine_Initialize(new Math.vec2(lX, lY), new Math.vec2(lWidth, lHeight), lUIConfiguration);
        // CppCall.Engine_Main(lUpdateDelegate, lRenderSceneDelegate, lRenderUIDelegate);
        // CppCall.Engine_Shutdown();
    }

    static void Main(string[] args)
    {
        Parser.Default.ParseArguments<Options>(args).WithParsed<Options>(GuardedMain);
    }
}