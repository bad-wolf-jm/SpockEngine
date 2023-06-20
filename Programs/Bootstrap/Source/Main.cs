using System;
using System.IO;
using System.Linq;
// using System.Drawing;
using System.Drawing.Text;
// using System.Windows.Media;
using System.Collections.Generic;

using CommandLine;
using SpockEngine;

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

    private static List<string> GetFilesForFont(string fontName)
    {
        var fontNameToFiles = new Dictionary<string, List<string>>();

        foreach (var fontFile in Directory.GetFiles(Environment.GetFolderPath(Environment.SpecialFolder.Fonts)))
        {
            var fc = new PrivateFontCollection();

            if (File.Exists(fontFile))
                fc.AddFontFile(fontFile);

            if ((!fc.Families.Any()))
                continue;

            var name = fc.Families[0].Name;

            // If you care about bold, italic, etc, you can filter here.
            if (!fontNameToFiles.TryGetValue(name, out var files))
            {
                files = new List<string>();
                fontNameToFiles[name] = files;
            }

            files.Add(fontFile);
        }

        if (!fontNameToFiles.TryGetValue(fontName, out var result))
            return null;

        return result;
    }

    static void Main(string[] args)
    {
        var lExePath = System.Reflection.Assembly.GetExecutingAssembly().Location;

        var lUserHome = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        var lUserConfig = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);


        var lUIFont = GetFilesForFont("Segoe UI");
        foreach( var x in lUIFont)
            Console.WriteLine(x);

        Console.WriteLine(lUserHome);
        Console.WriteLine(lUserConfig);
        Console.WriteLine(lExePath);

        Parser
            .Default
            .ParseArguments<Options>(args)
            .WithParsed<Options>(o =>
            {
                var lAppConfigRoot = Path.Combine(lUserConfig, o.Application);

                var lAppConfig = Path.Combine(lAppConfigRoot, "Config");
                if (!Directory.Exists(lAppConfig))
                    Directory.CreateDirectory(lAppConfig);

                var lAppLogs = Path.Combine(lAppConfigRoot, "Logs");
                if (!Directory.Exists(lAppLogs))
                    Directory.CreateDirectory(lAppLogs);

                var lAppUIConfig = Path.Combine(lAppConfig, "imgui.ini");
                var lAppIniConfig = Path.Combine(lAppConfig, "Application.yaml");
            });

    }
}