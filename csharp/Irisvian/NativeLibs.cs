using System;
using System.Runtime.InteropServices;

namespace Glasssix.Irisviel
{
    /// <summary>
    /// Defines available feaute models.
    /// </summary>
    public enum FeatureModel
    {
        Small,
        Large
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeDatabaseSearchResult
    {
        public IntPtr Record;
        public float Similarity;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeDatabaseRecordContent
    {
        public IntPtr Key;
        public IntPtr Feature;
        public UIntPtr KeySize;
        public UIntPtr FeatureSize;
    }

    /// <summary>
    /// An internal marshaler for return values that are "string".
    /// </summary>
    internal class IrisvielStringMarshaler : ICustomMarshaler
    {
        static ICustomMarshaler GetInstance(string cookie) => new IrisvielStringMarshaler();

        public void CleanUpManagedData(object managedObj)
        {
        }

        public void CleanUpNativeData(IntPtr nativeData) => NativeLibs.IrisvielFree(nativeData);

        public int GetNativeDataSize() => -1;

        public IntPtr MarshalManagedToNative(object managedObj) => throw new NotImplementedException();

        public object MarshalNativeToManaged(IntPtr nativeData)
        {
            return nativeData != null ? Marshal.PtrToStringAnsi(nativeData) : null;
        }
    }

    internal class NativeLibs
    {
        public const string DLL_NAME = "Irisviel";

        /// <summary>
        /// Creates an instance of the face searching module.
        /// </summary>
        /// <param name="singleDatabaseCapacity">The maximal number of items that one single database file can accomodate</param>
        /// <param name="dimension">The dimension of the feature</param>
        /// <param name="workingDirectory">The working directory in which the module creates internal files</param>
        /// <returns>The handle of the instance</returns>
        [DllImport(DLL_NAME, EntryPoint = "irisviel_create_instance")]
        public static extern IntPtr IrisvielCreateInstance(int singleDatabaseCapacity, int dimension, string workingDirectory);

        /// <summary>
        /// Creates a database record.
        /// </summary>
        /// <param name="model">The feature model</param>
        /// <returns>The handle of the record</returns>
        [DllImport(DLL_NAME, EntryPoint = "irisviel_create_record")]
        public static extern IntPtr IrisivelCreateRecord(FeatureModel model);

        /// <summary>
        /// Creates a database record with arguments.
        /// </summary>
        /// <param name="model">The feature model</param>
        /// <param name="key">The key</param>
        /// <param name="feature">The feature</param>
        /// <returns>The handle of the record</returns>
        [DllImport(DLL_NAME, EntryPoint = "irisviel_create_record_with_arguments", CharSet = CharSet.Ansi)]
        public static extern IntPtr IrisivelCreateRecordWithArguments(
            FeatureModel model,
            [MarshalAs(UnmanagedType.LPStr)] string key,
            [MarshalAs(UnmanagedType.LPArray)] float[] feature
            );

        /// <summary>
        /// Frees a memory block.
        /// </summary>
        /// <param name="memory">The memory pointer</param>
        [DllImport(DLL_NAME, EntryPoint = "irisviel_free")]
        public static extern void IrisvielFree(IntPtr memory);

        /// <summary>
        /// Frees an instance.
        /// </summary>
        /// <param name="instance">The handle of the instance</param>
        [DllImport(DLL_NAME, EntryPoint = "irisviel_free_instance")]
        public static extern void IrisvielFreeInstance(IntPtr instance);

        /// <summary>
        /// Frees a record.
        /// </summary>
        /// <param name="record">The handle of the record</param>
        [DllImport(DLL_NAME, EntryPoint = "irisviel_free_record")]
        public static extern void IrisvielFreeRecord(IntPtr record);

        /// <summary>
        /// Frees a record content.
        /// </summary>
        /// <param name="content">The content</param>
        [DllImport(DLL_NAME, EntryPoint = "irisviel_free_record_content")]
        public static extern void IrisvielFreeRecordContent(ref NativeDatabaseRecordContent content);

        /// <summary>
        /// Frees a searching result.
        /// </summary>
        /// <param name="result">The handle of the result</param>
        /// <param name="size">The size of the result</param>
        [DllImport(DLL_NAME, EntryPoint = "irisviel_free_search_result")]
        public static extern void IrisvielFreeSearchResult(IntPtr result, UIntPtr size);

        /// <summary>
        /// Sets the content of a record.
        /// </summary>
        /// <param name="record">The handle of the record</param>
        /// <param name="content">The content</param>
        [DllImport(DLL_NAME, EntryPoint = "irisviel_set_record_content")]
        public static extern void IrisvielSetRecordContent(IntPtr record, ref NativeDatabaseRecordContent content);

        /// <summary>
        /// Gets the content of a record.
        /// </summary>
        /// <param name="record">The handle of the record</param>
        /// <param name="content">The content</param>
        [DllImport(DLL_NAME, EntryPoint = "irisviel_get_record_content")]
        public static extern void IrisvielGetRecordContent(IntPtr record, out NativeDatabaseRecordContent content);

        /// <summary>
        /// Clears the loaded databases and keeps all data on the disk still existing.
        /// </summary>
        /// <param name="instance">The handle of the instance</param>
        [DllImport(DLL_NAME, EntryPoint = "irisviel_clear")]
        public static extern void IrisvielClear(IntPtr instance);

        /// <summary>
        /// Removes all the loaded databases and deletes the files on the disk.
        /// </summary>
        /// <param name="instance">The handle of the instance</param>
        [DllImport(DLL_NAME, EntryPoint = "irisviel_remove_all")]
        public static extern void IrisvielRemoveAll(IntPtr instance);

        /// <summary>
        /// Gets the database directory.
        /// </summary>
        /// <param name="instance">The handle of the instance</param>
        /// <returns>The database directory</returns>
        [DllImport(DLL_NAME, EntryPoint = "irisviel_database_directory", CharSet = CharSet.Ansi)]
        [return: MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(IrisvielStringMarshaler))]
        public static extern string IrisvielDatabaseDirectory(IntPtr instance);

        /// <summary>
        /// Gets the cache directory.
        /// </summary>
        /// <param name="instance">The handle of the instance</param>
        /// <returns>The cache directory</returns>
        [DllImport(DLL_NAME, EntryPoint = "irisviel_cache_directory", CharSet = CharSet.Ansi)]
        [return: MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(IrisvielStringMarshaler))]
        public static extern string IrisvielCacheDirectory(IntPtr instance);

        /// <summary>
        /// Loads the database on the disk.
        /// </summary>
        /// <param name="instance">The handle of the instance</param>
        [DllImport(DLL_NAME, EntryPoint = "irisviel_load_databases")]
        public static extern void IrisvielLoadDatabases(IntPtr instance);

        /// <summary>
        /// Searches for results.
        /// </summary>
        /// <param name="instance">The handle of the instance</param>
        /// <param name="feature">The feature</param>
        /// <param name="top">The top number of the items to be taken from the databases</param>
        /// <param name="result">The result</param>
        /// <returns>The size of the result</returns>
        [DllImport(DLL_NAME, EntryPoint = "irisviel_search")]
        public static extern UIntPtr IrisvielSearch(
            IntPtr instance,
            [MarshalAs(UnmanagedType.LPArray)] float[] feature,
            int top,
            out IntPtr result
            );

        /// <summary>
        /// Adds a record.
        /// </summary>
        /// <param name="instance">The handle of the instance</param>
        /// <param name="record">The handle of the record</param>
        [DllImport(DLL_NAME, EntryPoint = "irisviel_add_record")]
        public static extern void IrisvielAddRecord(IntPtr instance, IntPtr record);

        /// <summary>
        /// Adds some records.
        /// </summary>
        /// <param name="instance">The handle of the instance</param>
        /// <param name="records">The handles of the records</param>
        /// <param name="size">The size of the records</param>
        [DllImport(DLL_NAME, EntryPoint = "irisviel_add_records")]
        public static extern void IrisvielAddRecords(
            IntPtr instance,
            [MarshalAs(UnmanagedType.LPArray)] IntPtr[] records,
            UIntPtr size
            );

        /// <summary>
        /// Removes a record.
        /// </summary>
        /// <param name="instance">The handle of the instance</param>
        /// <param name="key">The key</param>
        [DllImport(DLL_NAME, EntryPoint = "irisviel_remove_record", CharSet = CharSet.Ansi)]
        public static extern void IrisvielRemoveRecord(
            IntPtr instance,
            [MarshalAs(UnmanagedType.LPStr)] string key
            );

        /// <summary>
        /// Removes some records.
        /// </summary>
        /// <param name="instance">The handle of the instance</param>
        /// <param name="keys">The keys</param>
        /// <param name="size">The size of the keys</param>
        [DllImport(DLL_NAME, EntryPoint = "irisviel_remove_records")]
        public static extern void IrisvielRemoveRecords(
            IntPtr instance,
            [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] keys,
            UIntPtr size
            );

        /// <summary>
        /// Updates a record.
        /// </summary>
        /// <param name="instance">The handle of the instance</param>
        /// <param name="record">The handle of the record</param>
        [DllImport(DLL_NAME, EntryPoint = "irisviel_update_record")]
        public static extern void IrisvielUpdateRecord(IntPtr instance, IntPtr record);

        /// <summary>
        /// Updates some records.
        /// </summary>
        /// <param name="instance">The handle of the instance</param>
        /// <param name="records">The handle of the record</param>
        /// <param name="size">The size of the keys</param>
        [DllImport(DLL_NAME, EntryPoint = "irisviel_update_records")]
        public static extern void IrisvielUpdateRecords(
            IntPtr instance,
            [MarshalAs(UnmanagedType.LPArray)] IntPtr[] records,
            UIntPtr size
            );
    }
}
