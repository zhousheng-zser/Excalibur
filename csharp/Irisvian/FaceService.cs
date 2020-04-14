using System;
using System.Linq;
using System.Runtime.InteropServices;

namespace Glasssix.Irisviel
{
    /// <summary>
    /// A face service for quick searching and databasing.
    /// </summary>
    public class FaceService : IDisposable
    {
        private bool _disposed;
        private IntPtr _instance;

        /// <summary>
        /// Gets the feature model.
        /// </summary>
        public FeatureModel FeatureModel { get; }

        /// <summary>
        /// Gets the database directory.
        /// </summary>
        public string DatabaseDirectory => _disposed ? throw new ObjectDisposedException(GetType().Name) : NativeLibs.IrisvielDatabaseDirectory(_instance);

        /// <summary>
        /// Gets the cache directory.
        /// </summary>
        public string CacheDirectory => _disposed ? throw new ObjectDisposedException(GetType().Name) : NativeLibs.IrisvielCacheDirectory(_instance);

        /// <summary>
        /// Creates an instance.
        /// </summary>
        /// <param name="model">The feature model</param>
        /// <param name="singleDatabaseCapacity">The maximal number of items that one single database file can accomodate</param>
        /// <param name="workingDirectory">The working directory in which the module creates internal files</param>
        public FaceService(FeatureModel model, int singleDatabaseCapacity, string workingDirectory)
        {
            FeatureModel = model;
            _instance = NativeLibs.IrisvielCreateInstance(singleDatabaseCapacity, model == FeatureModel.Small ? 128 : 512, workingDirectory);
        }

        /// <summary>
        /// The finalizer.
        /// </summary>
        ~FaceService()
        {
            Dispose(false);
        }

        /// <summary>
        /// Clears the loaded databases and keeps all data on the disk still existing.
        /// </summary>
        public void Clear()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(GetType().Name);
            }

            NativeLibs.IrisvielClear(_instance);
        }

        /// <summary>
        /// Removes all the loaded databases and deletes the files on the disk.
        /// </summary>
        public void RemoveAll()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(GetType().Name);
            }

            NativeLibs.IrisvielRemoveAll(_instance);
        }

        /// <summary>
        /// Loads the database on the disk.
        /// </summary>
        public void LoadDatabases()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(GetType().Name);
            }

            NativeLibs.IrisvielLoadDatabases(_instance);
        }

        /// <summary>
        /// Searches for results.
        /// </summary>
        /// <param name="feature">The feature</param>
        /// <param name="top">The top number of the items to be taken from the databases</param>
        /// <returns>The result</returns>
        public DatabaseSearchResult[] Search(float[] feature, int top)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(GetType().Name);
            }

            var nativeSize = NativeLibs.IrisvielSearch(_instance, feature, top, out var nativeResult);
            var size = (int)nativeSize;
            var result = Enumerable
                .Range(0, size)
                .Select(i => Marshal.PtrToStructure<NativeDatabaseSearchResult>(nativeResult + i * Marshal.SizeOf<NativeDatabaseSearchResult>()))
                .Select(a => new { Content = GetNativeRecordContentCore(a.Record), Similarity = a.Similarity })
                .Select(a => new DatabaseSearchResult(GetRecordCoreAndFreeNative(a.Content), a.Similarity))
                .ToArray();

            NativeLibs.IrisvielFreeSearchResult(nativeResult, nativeSize);

            return result;
        }

        /// <summary>
        /// Adds a record.
        /// </summary>
        /// <param name="key">The key</param>
        /// <param name="feature">The feature</param>
        public void Add(string key, float[] feature)
        {
            Add(new DatabaseRecord(key, feature));
        }

        /// <summary>
        /// Adds a record.
        /// </summary>
        /// <param name="record">The record</param>
        public void Add(DatabaseRecord record)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(GetType().Name);
            }

            var nativeRecord = NativeLibs.IrisivelCreateRecordWithArguments(FeatureModel, record.Key, record.Feature);

            NativeLibs.IrisvielAddRecord(_instance, nativeRecord);
            NativeLibs.IrisvielFreeRecord(nativeRecord);
        }

        /// <summary>
        /// Adds some records.
        /// </summary>
        /// <param name="records">The records</param>
        public void Add(DatabaseRecord[] records)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(GetType().Name);
            }

            var nativeRecords = records
                .Select(a => NativeLibs.IrisivelCreateRecordWithArguments(FeatureModel, a.Key, a.Feature))
                .ToArray();

            NativeLibs.IrisvielAddRecords(_instance, nativeRecords, (UIntPtr)records.Length);
            Array.ForEach(nativeRecords, a => NativeLibs.IrisvielFreeRecord(a));
        }

        /// <summary>
        /// Removes a record.
        /// </summary>
        /// <param name="key">The key</param>
        public void Remove(string key)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(GetType().Name);
            }

            NativeLibs.IrisvielRemoveRecord(_instance, key);
        }

        /// <summary>
        /// Removes some records.
        /// </summary>
        /// <param name="keys">The keys</param>
        public void Remove(string[] keys)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(GetType().Name);
            }

            NativeLibs.IrisvielRemoveRecords(_instance, keys, (UIntPtr)keys.Length);
        }

        /// <summary>
        /// Updates a record.
        /// </summary>
        /// <param name="record">The record</param>
        public void Update(DatabaseRecord record)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(GetType().Name);
            }

            var nativeRecord = NativeLibs.IrisivelCreateRecordWithArguments(FeatureModel, record.Key, record.Feature);

            NativeLibs.IrisvielUpdateRecord(_instance, nativeRecord);
            NativeLibs.IrisvielFreeRecord(nativeRecord);
        }

        /// <summary>
        /// Updates some records.
        /// </summary>
        /// <param name="records">The records</param>
        public void Update(DatabaseRecord[] records)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(GetType().Name);
            }

            var nativeRecords = records
               .Select(a => NativeLibs.IrisivelCreateRecordWithArguments(FeatureModel, a.Key, a.Feature))
               .ToArray();

            NativeLibs.IrisvielUpdateRecords(_instance, nativeRecords, (UIntPtr)records.Length);
            Array.ForEach(nativeRecords, a => NativeLibs.IrisvielFreeRecord(a));
        }

        /// <summary>
        /// Disposes all managed and unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Disposes all managed and unmanaged resources. 
        /// </summary>
        /// <param name="disposing">Indicates whether disposing managed resources is scheduled</param>
        protected void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                }

                if (_instance != null)
                {
                    NativeLibs.IrisvielFreeInstance(_instance);
                    _instance = IntPtr.Zero;
                }

                _disposed = true;
            }
        }

        private static NativeDatabaseRecordContent GetNativeRecordContentCore(IntPtr record)
        {
            NativeLibs.IrisvielGetRecordContent(record, out var content);

            return content;
        }

        private static DatabaseRecord GetRecordCoreAndFreeNative(NativeDatabaseRecordContent content)
        {
            var key = Marshal.PtrToStringAnsi(content.Key, (int)content.KeySize);
            var feature = new float[(int)content.FeatureSize];

            Marshal.Copy(content.Feature, feature, 0, feature.Length);
            NativeLibs.IrisvielFreeRecordContent(ref content);

            return new DatabaseRecord(key, feature);
        }
    }
}
