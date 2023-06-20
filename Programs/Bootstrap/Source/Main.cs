using System;
using CommandLine;
using SpockEngine;

class SEBootstrap
{
    public class Options
    {
        [Option('v', "verbose", Required = false, HelpText = "Set output to verbose messages.")]
        public bool Verbose { get; set; }
    }

    static void Main(string[] args)
    {
        var lExePath = System.Reflection.Assembly.GetExecutingAssembly().Location;

        var lUserHome = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        var lUserConfig = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        var lFontsFolder = Environment.GetFolderPath(Environment.SpecialFolder.Fonts);

        Console.WriteLine(lUserHome);
        Console.WriteLine(lUserConfig);
        Console.WriteLine(lFontsFolder);
        Console.WriteLine(lExePath);

        Parser
            .Default
            .ParseArguments<Options>(args)
            .WithParsed<Options>(o =>
            {
                if (o.Verbose)
                {
                    Console.WriteLine($"Verbose output enabled. Current Arguments: -v {o.Verbose}");
                    Console.WriteLine("Quick Start Example! App is in Verbose mode!");
                }
                else
                {
                    Console.WriteLine($"Current Arguments: -v {o.Verbose}");
                    Console.WriteLine("Quick Start Example!");
                }
            });

    }
}