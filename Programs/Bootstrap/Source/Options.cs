using System;
using System.IO;
using System.Linq;
using System.Drawing;
using System.Drawing.Text;
using System.Windows.Media;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Reflection;

using CommandLine;

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
