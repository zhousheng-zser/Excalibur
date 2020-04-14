namespace Glasssix.Irisviel
{
    /// <summary>
    /// A search result.
    /// </summary>
    public class DatabaseSearchResult
    {
        /// <summary>
        /// The record.
        /// </summary>
        public DatabaseRecord Record { get; }

        /// <summary>
        /// The similarity.
        /// </summary>
        public float Similarity { get; }

        /// <summary>
        /// Create an instance.
        /// </summary>
        /// <param name="record">The record</param>
        /// <param name="similarity">The similarity</param>
        public DatabaseSearchResult(DatabaseRecord record, float similarity)
        {
            Record = record;
            Similarity = similarity;
        }
    }
}
