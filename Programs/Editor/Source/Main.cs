using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Collections.Generic;
using System.Xml.Serialization;

using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

using SpockEngine;
using SpockEngine.Math;

namespace SEEditor
{

    public class SEEditor : SEApplication
    {
        bool mRequestQuit = false;

        UIMaterialEditor mMaterialEditor = new UIMaterialEditor();
        public SEEditor() { }

        public override bool UpdateMenu()
        {
            //  try
            //  {
            //     mFileMenu.Update();
            //  }
            //  catch (Exception e)
            //  {
            //     Console.WriteLine(e);
            //  }

            return mRequestQuit;
        }

        public override void Initialize(string aConfigurationPath)
        {
            //  if (!File.Exists(aConfigurationPath))
            //  {
            //     var serializer = new SerializerBuilder().WithNamingConvention(CamelCaseNamingConvention.Instance).Build();
            //     var yaml = serializer.Serialize(new Configuration());

            //     using (var writer = new StreamWriter(aConfigurationPath))
            //     {
            //        writer.Write(yaml);
            //     }
            //  }

            //  using (var lReader = new StreamReader(aConfigurationPath))
            //  {
            //     var deserializer = new DeserializerBuilder()
            //         .WithNamingConvention(CamelCaseNamingConvention.Instance)
            //         .Build();

            //     try
            //     {
            //        mConfiguration = deserializer.Deserialize<Configuration>(lReader);
            //     }
            //     catch
            //     {
            //        mConfiguration = new Configuration();
            //     }
            //  }

            //  mConnectedModules = new UIConnectecModules();
            //  mConnectedModules.OnConnectionRequest = OpenOlmConnection;

            //  mWorkspace = new UIWorkspace();
            //  mWorkspaceForm = new UIForm();
            //  mWorkspaceForm.SetTitle("WORKSPACE");
            //  mWorkspaceForm.SetContent(mWorkspace);

            //  mRunScript = new UIRunScript();
            //  mRunTests = new UIRunTests();

            //  mFileMenu = new UIMenu("File");

            //  mFileMenu.AddAction("Open iOlm file...", "").OnTrigger(() =>
            //  {
            //     var lFileName = SelectFile("OLM Files (*.iolm)|*.iolm|OLX Files (*.iolm)|*.olx|All Files (*.*)|*.*");

            //     if (lFileName != "")
            //     {
            //        var lDocument = new UIIolmDocument(lFileName);
            //        mWorkspace.Add(lDocument);
            //     }
            //  });

            //  mFileMenu.AddAction("Open OTDR file...", "").OnTrigger(() =>
            //  {
            //     var lFileName = SelectFile("OTDR Files (*.trc)|*.trc|All Files (*.*)|*.*");

            //     if (lFileName != "")
            //     {
            //        var lDocument = new UIOtdrDocument(lFileName);
            //        mWorkspace.Add(lDocument);
            //     }
            //  });

            //  mFileMenu.AddAction("Open iOlm diff file...", "").OnTrigger(() =>
            //  {
            //     var lFileName = SelectFile("OLM Files (*.iolm)|*.iolm|OLX Files (*.iolm)|*.olx|All Files (*.*)|*.*");

            //     if (lFileName != "")
            //     {
            //        var lDocument = new UIIolmDiffDocument(lFileName);
            //        mWorkspace.Add(lDocument);
            //     }
            //  });

            //  mFileMenu.AddSeparator();

            //  mFileMenu.AddAction("Open test report...", "").OnTrigger(() =>
            //  {
            //     var lFileName = SelectFile("XML Files (*.xml)|*.xml|All Files (*.*)|*.*");

            //     if (lFileName != "")
            //     {
            //        var lTestFailFolder = Path.GetDirectoryName(lFileName);
            //        var lDocument = new UITestFailResultTable(Path.GetDirectoryName(lFileName));

            //        lDocument.OnElementClicked((string aTestName, string aMessage, string aFileName) =>
            //        {
            //           mWorkspace.Add(new UIIolmTestFailDocument(aTestName, aMessage, aFileName));
            //        });

            //        mWorkspace.Add(lDocument);
            //     }
            //  });

            //  mFileMenu.AddSeparator();

            //  mFileMenu.AddAction("Load unit test assembly...", "").OnTrigger(() =>
            //  {
            //     if (mUnitTestDomain != null)
            //     {
            //        AppDomain.Unload(mUnitTestDomain);
            //     }

            //     string pathToDll = @"D:\Build\Lib\debug\develop\OlmUnitTests\OlmUnitTests.dll";
            //     AppDomainSetup domainSetup = new AppDomainSetup { PrivateBinPath = pathToDll, ApplicationBase = ".", ConfigurationFile = "" };
            //     mUnitTestDomain = AppDomain.CreateDomain("UnitTestDomain", null, domainSetup);
            //     mUnitTestProxy = (IUnitTestProxy)(mUnitTestDomain.CreateInstanceFromAndUnwrap(pathToDll, "OlmUnitTests.OlmUnitTests"));
            //     mUnitTestProxy.Initialize("D:\\OTDR\\UnitTestOlm\\TestData", "d:\\AutomatedTests");

            //     mRunTests.SetAssemblyName(pathToDll);
            //     mRunTests.Configure(mUnitTestProxy);
            //     mRunTests.Open();

            //  });

            //  mFileMenu.AddSeparator();

            //  mFileMenu.AddAction("Quit", "").OnTrigger(() => { this.mRequestQuit = true; });
        }

        //   private void OpenOlmConnection(string aName, string aAddress)
        //   {
        //      var lConnection = new UIIolmDeviceDocument(aName, aAddress);
        //      mDeviceConnections.Add(lConnection);
        //      mWorkspace.Add(lConnection);
        //   }

        //   private string SelectFile(string aFilter)
        //   {
        //      return CppCall.OpenFile(aFilter);
        //   }

        //   public override void Shutdown(string aConfigurationPath)
        //   {
        //      var serializer = new SerializerBuilder().WithNamingConvention(CamelCaseNamingConvention.Instance).Build();
        //      var yaml = serializer.Serialize(mConfiguration);

        //      using (var writer = new StreamWriter(aConfigurationPath))
        //      {
        //         writer.Write(yaml);
        //      }
        //   }

        public override void Update(float aTs)
        {
            try
            {
                mMaterialEditor.Update();
                // mWorkspaceForm.Update();
                // mConnectedModules.Update();

                // mRunScript.Tick(aTs);
                // mRunScript.Update();

                // mRunTests.Tick(aTs);
                // mRunTests.Update();

                // foreach (var c in mDeviceConnections)
                //     c.Tick(aTs);
            }
            catch (Exception e)
            {
                Console.WriteLine(e);
            }

        }

        public override void UpdateUI(float aTs)
        {
        }
    }
}