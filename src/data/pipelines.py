from langdetect import detect
import re
from sklearn.base import TransformerMixin


class ColumnsFilter(TransformerMixin):
    """
    Transformer to drop columns of the dataframe.

    :param columns: list, list of columns to drop.
    :param except_columns: list, list of columns to preserve.
    """

    def __init__(self, columns=None, all_except=None):
        self.columns = columns
        self.all_except = all_except

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df, **transform_params):
        if self.columns is not None:
            columns = self.columns
        elif self.all_except is not None and len(self.all_except):
            columns = df[df.columns.difference(self.all_except)]
        else:
            return df

        df = df.drop(columns, axis=1)
        return df


class NanFilter(TransformerMixin):
    """
    Filter NaN values in dataset of selected columns.

    :param columns: list, subset of columns to drop samples with NaN
        value in.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df, **transform_params):
        df = df.dropna(subset=self.columns)
        return df


class ArticlesLanguageFilter(TransformerMixin):
    """
    Filter to remove articles written in different language.

    :param column: str, name of column to check the language in.
    :param language: str, shortcut of language (e.g. 'en').
    """

    def __init__(self, column, language):
        self.column = column
        self.language = language

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df, **transform_params):
        df_copy = df.copy()

        df_copy['lang'] = df_copy[self.column].apply(lambda text: detect(text))
        df_copy = df_copy[df_copy.lang == self.language]

        df_copy.drop(['lang'], axis=1, inplace=True)

        return df_copy


class ArticlesSizeFilter(TransformerMixin):
    """
    Filter to remove all articles that are too short or too long.

    :param column: str, name of column to check the size of article.
    :param upper_boundary: int, maximum words in article.
    :param lower_boundary: int, minimum words in article.
    """

    def __init__(self, column, lower_boundary, upper_boundary):
        self.column = column
        self.lower_boundary = lower_boundary
        self.upper_boundary = upper_boundary

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df, **transform_params):
        df_copy = df.copy()

        df_copy['num_words'] = df_copy[self.column].apply(
            lambda text: len(text.split())
        )
        df_copy = df_copy[
            (df_copy['num_words'] >= self.lower_boundary) &
            (df_copy['num_words'] <= self.upper_boundary)
        ]

        df_copy.drop(['num_words'], axis=1, inplace=True)

        return df_copy


class TextPreprocessor(TransformerMixin):
    """
    Transformer to clean text attribute.

    Following transformations are included:
    1. Making text lowercase.
    2. Removing html tags.
    3. Remove special characters.

    :param column: str, name of column to clean text in.
    """

    def __init__(self, column):
        self.column = column

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df, **transform_params):
        df_copy = df.copy()

        # Lowercase text.
        df_copy[self.column] = df_copy[self.column].apply(
            lambda text: text.lower()
        )

        # Remove html characters.
        df_copy[self.column] = df_copy[self.column].apply(
            lambda text: re.sub(r'<.*?>', '', text)
        )
        # Remove other special characters.
        df_copy[self.column] = df_copy[self.column].apply(
            lambda text: re.sub(r'[^a-zA-Z0-9\.,?!]+', ' ', text)
        )

        return df_copy
