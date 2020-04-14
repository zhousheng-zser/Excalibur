namespace Glasssix.Irisviel
{
    /// <summary>
    /// A database record.
    /// </summary>
    public class DatabaseRecord
    {
        /// <summary>
        /// The key.
        /// </summary>
        public string Key { get; set; }

        /// <summary>
        /// The feature.
        /// </summary>
        public float[] Feature { get; set; }

        /// <summary>
        /// Create an instance.
        /// </summary>
        public DatabaseRecord()
        {
        }

        /// <summary>
        /// Create an instance.
        /// </summary>
        /// <param name="key">The key</param>
        /// <param name="feature">The feature</param>
        public DatabaseRecord(string key, float[] feature)
        {
            Key = key;
            Feature = feature;
        }
    }
}
