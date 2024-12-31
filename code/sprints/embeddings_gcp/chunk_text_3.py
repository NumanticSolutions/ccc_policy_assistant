# [2412] n8
#

import pandas as pd

class ChunkText:

    def chunk_string(self, s, word_count):

        """Generate fixed word count chunks of the provided string.

        This generator yields consecutive substrings from 'string' with each substring
        having a length equal to word_count. The last chunk may be shorter than word_count if 
        the string's length is not divisible by word_count.

        Parameters:
            string (str)     : The input text that needs to be divided into chunks.
            word_count (int) : The desired word count size for each chunk, except possibly the last one.

        Yields:
            chunks (list strs)  : The chunked text as a list of strings.
        """

        chunks = []
        ts = s.split()
        i = 0
        while i < len(ts):
            chunk = ''
            for j in range(word_count):
                chunk += ts[i] + " "
                i += 1
                if i == len(ts):
                    break
            chunks.append(chunk)

        return chunks       

    def chunk_dataframe(self, df, col_txt = 'ptag_text', col_meta = 'url', chunk_word_count = 150):

        """ Chunk a DataFrame text column into smaller pieces based on word count 
        and collect corresponding metadata.

        Parameters:
            df (dataframe)   : The input pandas dataframe
            col_txt (str)    : Name of the column with the text to be chunked
            col_meta (str)   : Name of the column with the corresponding metadata
            word_count (int) : The desired word count size for each chunk, except possibly the last one.

        Yields:
            chunks (list strs)  : The chunked text as a list of strings.
            meta (list strs)    : The corresponding metadata for each chunk, ie source urls

        """

        chunks, metas = [], []
        for i, row in df.iterrows():
            meta = row[col_meta]
            text = row[col_txt]
            chunked = self.chunk_string(text, chunk_word_count)
            for chunk in chunked:
                metas.append(meta)
                chunks.append(chunk)

        return chunks, metas

