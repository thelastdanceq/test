import sys
sys.path.insert(0, "../")
sys.path.insert(0, "./")
from tidal_analysis import *

class TestDataManipalations():
    def test_get_longest_contiguous_data(self):
        data = {
            'Time': pd.to_datetime(['2020-01-01 00:00', '2020-01-01 01:00', '2020-01-01 02:00',
                                    '2020-01-02 00:00', '2020-01-02 01:00', '2020-01-02 02:00',
                                    '2020-01-03 00:00', '2020-01-03 01:00', '2020-01-03 02:00',
                                    '2020-01-03 09:00'
                                    ]),
            'Sea Level': [1.0, 2.0, np.nan, 4.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        }
        df = pd.DataFrame(data)

        result = get_longest_contiguous_data(df)
        result.reset_index(drop=True, inplace=True)


        expected_dates = pd.to_datetime(
            ['2020-01-01 00:00', '2020-01-01 01:00'])

        pd.testing.assert_series_equal(
            result['Time'], pd.Series(expected_dates, name='Time'))
