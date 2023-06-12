using System;
using System.Reflection;
using System.Runtime;
using System.Collections.Generic;
using System.Linq;

namespace SpockEngine
{
    public static class Utilities
    {
        public static IEnumerable<Type> GetAllDerivedTypes<T>(bool aIncludeAllAssemblies = true)
        {
            var lType = typeof(T);

            List<Type> lDerivedTypes = GetAllDerivedTypes<T>(Assembly.GetAssembly(lType)).ToList();

            if (aIncludeAllAssemblies)
            {
                IEnumerable<Assembly> lDependentAssemblies = GetDependentAssemblies(Assembly.GetAssembly(lType));

                lDerivedTypes.AddRange(lDependentAssemblies.SelectMany(a => GetAllDerivedTypes<T>(a)));
            }

            return lDerivedTypes;
        }

        private static IEnumerable<Assembly> GetDependentAssemblies(Assembly aAnalyzedAssembly)
        {
            return AppDomain.CurrentDomain.GetAssemblies().Where(aAssembly =>
            {
                return aAssembly.GetReferencedAssemblies()
                                .Select(a => a.FullName)
                                .Contains(aAnalyzedAssembly.FullName);
            });
        }

        private static IEnumerable<Type> GetAllDerivedTypes<T>(Assembly aAssembly)
        {
            var lBaseType = typeof(T);
            TypeInfo lBaseTypeInfo = lBaseType.GetTypeInfo();

            return aAssembly.DefinedTypes.Where(aType =>
            {
                if (lBaseTypeInfo.IsClass)
                    return aType.IsSubclassOf(lBaseType);

                return lBaseTypeInfo.IsInterface && aType.ImplementedInterfaces.Contains(lBaseTypeInfo.AsType());

            }).Select(aType => aType.AsType());
        }

        public static T GetProperty<T>(object aObject, string aName)
        {
            if (aObject == null) return default(T);

            var lObjectType = aObject.GetType();
            var lProperty = lObjectType.GetProperty(aName);

            return (T)lProperty.GetValue(aObject);
        }
    }
}
