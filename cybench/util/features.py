import numpy as np
import pandas as pd
from datetime import datetime

from cybench.config import (
    KEY_LOC,
    KEY_YEAR,
    KEY_DATES,
    GDD_BASE_TEMP,
    GDD_UPPER_LIMIT,
)


def fortnight_from_date(dt: datetime):
    """Get the fortnight number from date.

    Args:
      dt: date

    Returns:
      Fortnight number, "YYYY0101" to "YYYY0115" -> 1.
    """
    month = dt.month
    day_of_month = dt.day
    fortnight_number = (month - 1) * 2
    if day_of_month <= 15:
        return fortnight_number + 1
    else:
        return fortnight_number + 2


def dekad_from_date(dt: datetime):
    """Get the dekad number from date.

    Args:
      dt: date

    Returns:
      Dekad number, e.g. "YYYY0101" to "YYYY0110" -> 1,
                         "YYYY0111" to "YYYY0120" -> 2,
                         "YYYY0121" to "YYYY0131" -> 3
    """
    month = dt.month
    day_of_month = dt.day
    dekad = (month - 1) * 3
    if day_of_month <= 10:
        dekad += 1
    elif day_of_month <= 20:
        dekad += 2
    else:
        dekad += 3

    return dekad


def _add_period(df: pd.DataFrame, period_length: str):
    """Add a period column.

    Args:
      df : pd.DataFrame
      period_length: string, which can be "month", "fortnight" or "dekad"

    Returns:
      pd.DataFrame
    """
    # NOTE expects data column in string format
    # add a period column based on time step
    if period_length == "month":
        df["period"] = df["date"].dt.month
    elif period_length == "fortnight":
        df["period"] = df.apply(lambda r: fortnight_from_date(r["date"]), axis=1)
    elif period_length == "dekad":
        df["period"] = df.apply(lambda r: dekad_from_date(r["date"]), axis=1)

    return df


# Period can be a month or fortnight (biweekly or two weeks)
# Period sum of TAVG, TMIN, TMAX, PREC
def _aggregate_by_period(
    df: pd.DataFrame, index_cols: list, period_col: str, aggrs: dict, ft_cols: dict
):
    """Aggregate data into features by period.

    Args:
      df : pd.DataFrame
      index_cols: list of indices, which are location and year
      period_col: string, column added by add_period()
      aggrs: dict containing columns to aggregate (keys) and corresponding
             aggregation function (values)
      ft_cols: dict for renaming columns to feature columns

    Returns:
      pd.DataFrame with features
    """
    groupby_cols = index_cols + [period_col]
    ft_df = df.groupby(groupby_cols, observed=True).agg(aggrs).reset_index()

    # rename to indicate aggregation
    ft_df = ft_df.rename(columns=ft_cols)

    # pivot to add a feature column for each period
    ft_df = (
        ft_df.pivot_table(index=index_cols, columns=period_col, values=ft_cols.values())
        .fillna(0.0)
        .reset_index()
    )

    # combine names of two column levels
    # second level is period number
    ft_df.columns = [first + str(second) for first, second in ft_df.columns]

    return ft_df


# Vernalization requirement
# NOTE: Not using vernalization for the neurips submission
# def _calculate_vernalization(tavg):

#    def vrn_fac(temp):
#        if temp < 0:
#            return 0
#        elif 0 <= temp <= 4:
#            return temp / 4
#        elif 4 < temp <= 8:
#            return 1
#        elif 8 < temp <= 10:
#            return (10 - temp) / 2
#        else:
#            return 0

#    v_units = np.vectorize(vrn_fac)(tavg)

#    total_v_units = v_units.sum()

#    return total_v_units


def _count_threshold(
    df: pd.DataFrame,
    index_cols: list,
    period_col: str,
    indicator: str,
    threshold_exceed: bool = True,
    threshold: float = 0.0,
    ft_name: str = None,
):
    """Aggregate data into features by period.

    Args:
      df : pd.DataFrame
      index_cols: list of indices, which are location and year
      period_col: string, column added by add_period()
      indicator: string, indicator column to aggregate
      threshold_exceed: boolean
      threshold: float

      ft_name: string name for aggregated indicator

    Returns:
      pd.DataFrame with features
    """
    groupby_cols = index_cols + [period_col]
    if threshold_exceed:
        threshold_lambda = lambda x: 1 if (x[indicator] > threshold) else 0
    else:
        threshold_lambda = lambda x: 1 if (x[indicator] < threshold) else 0

    df["meet_thresh"] = df.apply(threshold_lambda, axis=1)
    ft_df = (
        df.groupby(groupby_cols, observed=True)
        .agg(FEATURE=("meet_thresh", "sum"))
        .reset_index()
    )
    # drop the column we added
    df = df.drop(columns=["meet_thresh"])

    if ft_name is not None:
        ft_df = ft_df.rename(columns={"FEATURE": ft_name})

    # pivot to add a feature column for each period
    ft_df = (
        ft_df.pivot_table(index=index_cols, columns=period_col, values=ft_name)
        .fillna(0.0)
        .reset_index()
    )

    # rename period cols
    period_cols = df["period"].unique()
    rename_cols = {p: ft_name + "p" + str(p) for p in period_cols}
    ft_df = ft_df.rename(columns=rename_cols)

    return ft_df


def unpack_time_series(df: pd.DataFrame, indicators: list):
    """Unpack time series data to rows per date.

    Args:
      df : pd.DataFrame

      indicators: list of indicators to unpack

    Returns:
      pd.DataFrame
    """
    # If indicators are not in the dataframe
    if set(indicators).intersection(set(df.columns)) != set(indicators):
        return None

    # for a data source, dates should match across all indicators
    df["date"] = df.apply(lambda r: r[KEY_DATES][indicators[0]], axis=1)

    # explode time series columns and dates
    df = df.explode(indicators + ["date"]).drop(columns=[KEY_DATES])

    return df

###############################################
# C.J. (11/2025): Adding more feature engineering

def assign_aez_from_latlon(df,
                           lat_col='lat',
                           lon_col='lon'):
    """
    Estimate AEZ from latitude & longitude only.

    Returns a new column `result_col` containing one of:
      'equatorial', 'tropical', 'subtropical', 'temperate',
      'boreal', 'polar', 'arid-hot', 'arid-cold', 'mediterranean'
    """
    # store coordinates
    lat = df[lat_col].to_numpy(dtype=float)
    lon = df[lon_col].to_numpy(dtype=float)
    lat_abs = np.abs(lat)

    # start with coarse latitudinal belt
    aez = np.full(lat.shape, "unknown", dtype=object)
    # Equatorial: very near equator
    aez[lat_abs <= 5.0] = "equatorial"
    # Tropical: tropical belt up to Tropic of Cancer/Capricorn
    mask = (lat_abs > 5.0) & (lat_abs <= 23.5)
    aez[mask] = "tropical"
    # Subtropical: between tropics and mid-latitudes
    mask = (lat_abs > 23.5) & (lat_abs <= 35.0)
    aez[mask] = "subtropical"
    # Temperate / warm-temperate
    mask = (lat_abs > 35.0) & (lat_abs <= 50.0)
    aez[mask] = "temperate"
    # Boreal / cold-temperate
    mask = (lat_abs > 50.0) & (lat_abs <= 66.5)
    aez[mask] = "boreal"
    # Polar
    aez[lat_abs > 66.5] = "polar"

    # ----- override rules for major arid regions (hot deserts) -----
    # Each desert is approximated by a simple lat/lon bounding box (coarse).
    # Boxes: (min_lat, max_lat, min_lon, max_lon) in degrees.
    # NOTE: longitudes assumed to be -180..+180
    desert_boxes = [
        # Sahara (N Africa)
        (15.0, 35.0, -18.0, 35.0),
        # Arabian Desert / Arabian Peninsula
        (15.0, 32.0, 30.0, 60.0),
        # Thar Desert (India/Pakistan)
        (22.0, 31.0, 68.0, 78.0),
        # Australian interior (Great Sandy/Gibson/Simpson)
        (-35.0, -15.0, 113.0, 153.0),
        # Sonoran/Chihuahuan + SW USA / N Mexico deserts
        (20.0, 35.0, -125.0, -100.0),
        # Atacama (coastal Chile)
        (-30.0, -20.0, -75.0, -68.0),
        # Namib (SW Africa)
        (-28.0, -16.0, 11.0, 25.0)
    ]
    # apply hot-arid override if point in any desert box and latitude suggests warm
    is_hot_arid = np.zeros(lat.shape, dtype=bool)
    for (min_lat, max_lat, min_lon, max_lon) in desert_boxes:
        in_lat = (lat >= min_lat) & (lat <= max_lat)
        in_lon = (lon >= min_lon) & (lon <= max_lon)
        is_hot_arid |= (in_lat & in_lon)
    # label hot-arid only if lat_abs <= ~40 (to avoid classifying cold deserts like Gobi as hot)
    aez[is_hot_arid & (lat_abs <= 40.0)] = "arid-hot"

    # ----- override for cold/arid regions (e.g., Gobi, central Asia) -----
    cold_arid_boxes = [
        (40.0, 50.0, 85.0, 120.0),   # Gobi / Mongolia / N China
        (35.0, 55.0, 40.0, 70.0),    # central Asian arid belt (parts of Kazakhstan/Uzbek/etc)
    ]
    is_cold_arid = np.zeros(lat.shape, dtype=bool)
    for (min_lat, max_lat, min_lon, max_lon) in cold_arid_boxes:
        in_lat = (lat >= min_lat) & (lat <= max_lat)
        in_lon = (lon >= min_lon) & (lon <= max_lon)
        is_cold_arid |= (in_lat & in_lon)
    aez[is_cold_arid] = "arid-cold"

    # ----- Mediterranean climate (warm, dry summer, wet winter) approximation -----
    # Rough bounding box around Mediterranean basin
    med_mask = (lat >= 30.0) & (lat <= 45.0) & (lon >= -10.0) & (lon <= 40.0)
    aez[med_mask] = "mediterranean"

    # ----- final cleanup: fallback for any remaining Unknown values -----
    aez[aez == "unknown"] = "temperate"  # conservative fallback

    # return aez as new column vector
    return aez

def assign_aez_from_climate(df,
                            lat_col='lat',
                            tmean_col='temp_mean',
                            precip_sum_col='precip_sum'):
    """
    Estimate Agro-Ecological Zone (AEZ) from temperature and precipitation.
    Based on FAO/GAEZ simplified climatic classification.
    """

    # --- extract numeric arrays ---
    lat = df[lat_col].to_numpy(dtype=float)
    tmean = df[tmean_col].to_numpy(dtype=float)
    precip = df[precip_sum_col].to_numpy(dtype=float)

    # --- Step 1: thermal zones ---
    thermal_zone = np.empty_like(tmean, dtype=object)
    thermal_zone[tmean < 0] = "polar"
    thermal_zone[(tmean >= 0) & (tmean < 5)] = "boreal"
    thermal_zone[(tmean >= 5) & (tmean < 10)] = "cool_temperate"
    thermal_zone[(tmean >= 10) & (tmean < 18)] = "warm_temperate"
    thermal_zone[(tmean >= 18) & (tmean < 24)] = "subtropical"
    thermal_zone[tmean >= 24] = "tropical"

    # --- Step 2: moisture regimes ---
    moisture_zone = np.empty_like(precip, dtype=object)
    moisture_zone[precip < 300] = "arid"
    moisture_zone[(precip >= 300) & (precip < 700)] = "semi-arid"
    moisture_zone[(precip >= 700) & (precip < 1200)] = "sub-humid"
    moisture_zone[precip >= 1200] = "humid"

    # --- Step 3: combine ---
    aez = thermal_zone + "_" + moisture_zone

    # --- Step 4: refine based on latitude (polar override) ---
    polar_mask = np.abs(lat) > 66.5
    aez[polar_mask] = "polar"

    return aez

def arrhenius_response(Tleaf, Ea=60000.0, TrefK=298.15, R=8.314):
    '''
    Temperature dependence approximation for biochemical rates
    Arrhenius function: In chemical kinetics the rate constant k of
    many elementary reactions increases approximately exponentially
    with absolute temperature, classically expressed as:
      k(T) = A*exp(-Ea/(R*T))
          A = pre-exponential factor
          Ea = activation energy (J/mol)
          R = universal gas constant 8.314 (J/mol/K)
          T = temperature in Kelvin
    Arrhenius can be used to scale Rubisco carboxylation capacity Vcmax,
    electron-transport capacity Jmax, respiration rates, etc. from a
    reference temperature (TrefK, usually 25C = 298.15K) to leaf temperature Tleaf (°C).
    Here we use the ratio k(Tleaf)/k(TrefK) to scale rates.
    Reference:
     Carl J. Bernacchi, Archie R. Portis, Hiromi Nakano, Susanne von Caemmerer, Stephen P. Long (2002):
     Temperature Response of Mesophyll Conductance. Implications for the Determination of Rubisco Enzyme Kinetics and for Limitations to Photosynthesis in Vivo.
     Plant Physiology, Volume 130, Issue 4. https://doi.org/10.1104/pp.008250
    '''
    Tk = Tleaf + 273.15
    return np.exp((Ea / R) * (1.0 / TrefK - 1.0 / Tk))

def compute_farquhar_proxy(Tmean, Psum, Rsum, N=0.1, co2=400.0):
    '''
    Compute Farquhar-based seasonal proxy for photosynthesis rate.
    The derived biophysically interpretable proxy approximates the seasonal
    photosynthesis behavior as kind of a seasonal summary, not a simulator.
    - Converts pr and rsds units internally to mm/day and MJ/m2/day.
    - Calculate mean temperature and PAR from rsds (48% of shortwave radiation).
    - Returns a numpy array (one value per input row).
    '''

    # Unit conversions
    # pr: kg m-2 s-1 -> mm/day (1 kg m-2 = 1 mm water)
    #pr[pr_days] = pr[pr_days] * 86400.0
    # rsds: W m-2 -> MJ m-2 day-1
    #rsds[rsds_days] = rsds[rsds_days] * (86400.0 / 1e6)

    # Mean temps and PAR
    #Tmean = tas[tas_days].mean(axis=1).values
    PAR = 0.48 * Rsum
    #PAR = 0.48 * rsds[rsds_days].sum(axis=1).values  # approx fraction of shortwave as PAR (upper limit)
    # Reference: https://doi.org/10.1016/j.jag.2022.102724

    # soil proxies: nitrogen and co2 - fill sensible defaults where missing
    #N = N.fillna(0.1).values if 'nitrogen' in soil.columns else np.zeros(len(soil))
    nitrogen = np.full(len(Tmean), N)  # assume moderate N level if missing
    #co2 = co2.fillna(400.0).values if 'co2' in soil.columns else np.full(len(soil), 400.0) # use 400 ppm as default
    ambient_co2 = np.full(len(Tmean), co2)  # assume ambient CO2 if missing

    # Vcmax & J proxies
    temp_resp = arrhenius_response(Tmean) # Both respond exponentially to temperature
    Vcmax = 100.0 * temp_resp * (1.0 + 5.0 * nitrogen) # Rubisco capacity scales with leaf nitrogen (Rubisco is N-rich)
    # scaling factor 100, linear relationship of higher N -> higher Rubisco activity -> higher Vcmax (References: https://doi.org/10.1093/jxb/44.5.907, Farquhar 1980)
    J = 0.1 * PAR * temp_resp # Electron transport capacity scales with PAR

    # intercellular CO2 proxy and biochemical scalars
    ci = 0.7 * ambient_co2 # simplification for C3 plants under moderate water stress (upper limit), Reference: https://doi.org/10.1104/pp.111.1.179
    Kc = 404.9 # Michaelis–Menten constant for Rubisco with respect to CO2, Reference: https://doi.org/10.1111/j.1365-3040.2001.00668.x
    Gamma = 42.75 # CO2 compensation point in absence of mitochondrial respiration, Reference: https://doi.org/10.1111/j.1365-3040.2001.00668.x
    # Based on original Farquhar model for C3 photosynthesis:
    Ac = Vcmax * (ci - Gamma) / (ci + Kc + 1e-9)
    Aj = (J * (ci - Gamma)) / (4.0 * ci + 8.0 * Gamma + 1e-9)

    # harmonic-like mean approximating limitation by Ac or Aj
    A_proxy = (2.0 * Ac * Aj) / (Ac + Aj + 1e-9)

    # water limitation (saturating)
    #Psum = pr[pr_days].sum(axis=1).values
    water_lim = Psum / (Psum + 300.0)

    # final scaling by PAR availability and water limitation
    A_final = A_proxy * (PAR / (PAR + 100.0)) * water_lim
    A_final = np.nan_to_num(A_final, nan=0.0, posinf=0.0, neginf=0.0)
    return A_final

def new_features_cj():
    """Aggregate season-level features and return a DataFrame aligned by ID.
    This function enforces index->ID conversion and unit corrections.
    """

    # Convert units once per season-level computation
    pr[pr_days] = pr[pr_days] * 86400.0  # mm/day
    rsds[rsds_days] = rsds[rsds_days] * (86400.0 / 1e6)  # MJ/m2/day

    out = pd.DataFrame({'ID': pr['ID']}).astype('int32')
    # seasonal features from climate input
    out['Psum'] = pr[pr_days].sum(axis=1).astype('float32') # precipitation sum over all days
    out['Psd'] = pr[pr_days].std(axis=1).astype('float32')
    out['RadSum'] = rsds[rsds_days].sum(axis=1).astype('float32') # radiation sum over all days
    out['RadSd'] = rsds[rsds_days].std(axis=1).astype('float32')
    out['Tmean'] = tas[tas_days].mean(axis=1).astype('float32') # mean daily temperature averaged over all days
    out['Tsum'] = tas[tas_days].sum(axis=1).astype('float32') # mean daily temperature summed over all days
    out['Tsd'] = tas[tas_days].std(axis=1).astype('float32')
    out['Tmax'] = tmax[tmax_days].max(axis=1).astype('float32') # max daily temperature averaged over all days
    out['Tmin'] = tmin[tmin_days].min(axis=1).astype('float32') # min daily temperature averaged over all days
    out['Tmax_mean'] = tmax[tmax_days].mean(axis=1).astype('float32')
    out['Tmax_sd'] = tmax[tmax_days].std(axis=1).astype('float32')
    out['Tmin_mean'] = tmin[tmin_days].mean(axis=1).astype('float32')
    out['Tmin_sd'] = tmin[tmin_days].std(axis=1).astype('float32')

    # sub-seasonal (s1) features (can be extended to more windows)
    out['Psum_s1'] = pr[pr_days[0:30]].sum(axis=1).astype('float32') 
    out['Psd_s1'] = pr[pr_days[0:30]].std(axis=1).astype('float32')
    out['RadSum_s1'] = rsds[rsds_days[0:30]].sum(axis=1).astype('float32') 
    out['RadSd_s1'] = rsds[rsds_days[0:30]].std(axis=1).astype('float32')
    out['Tmean_s1'] = tas[tas_days[0:30]].mean(axis=1).astype('float32') 
    out['Tsum_s1'] = tas[tas_days[0:30]].sum(axis=1).astype('float32')
    out['Tsd_s1'] = tas[tas_days[0:30]].std(axis=1).astype('float32')
    out['Tmax_s1'] = tmax[tmax_days[0:30]].max(axis=1).astype('float32') 
    out['Tmin_s1'] = tmin[tmin_days[0:30]].min(axis=1).astype('float32') 
    out['Tmax_mean_s1'] = tmax[tmax_days[0:30]].mean(axis=1).astype('float32')
    out['Tmax_sd_s1'] = tmax[tmax_days[0:30]].std(axis=1).astype('float32')
    out['Tmin_mean_s1'] = tmin[tmin_days[0:30]].mean(axis=1).astype('float32')
    out['Tmin_sd_s1'] = tmin[tmin_days[0:30]].std(axis=1).astype('float32')
    
    # Farquhar proxy (calls internal conversions)
    out['Photo_Farquhar_approx'] = compute_farquhar_proxy(Tmean, Psum, Rsum, N, co2).astype('float32') # Farquhar photosynthesis proxy
    out['N_Co2_interact'] = N.fillna(0.1).values * co2.fillna(400.0).values
    out['N_Co2_ratio'] = N.fillna(0.1).values / co2.fillna(400.0).values

    # add rough estimation of Agroecological Zone (AEZ)
    out['aez'] = assign_aez_from_latlon(out, lat_col='lat', lon_col='lon')
    # add AEZ based on mean annual temperature and precipitation sum
    out['aez_clim'] = assign_aez_from_climate(out, lat_col='lat', tmean_col='Tmean', precip_sum_col='Psum')
    
    # add crop name
    out['crop'] = crop
    return 0
##### END C.J. #################################

# LOC, YEAR, DATE => cumsum by month
def growing_degree_days(df: pd.DataFrame, tbase: float):
    # Base temp would be 0 for winter wheat and 10 for corn.
    gdd = np.maximum(0, df["tavg"] - tbase)

    return gdd.sum()


def design_features(
    crop: str,
    input_dfs: dict,
):
    """Design features based domain expertise.

    Args:
      crop (str): crop name, e.g. maize
      input_dfs (dict): keys are input names, values are pd.DataFrames

    Returns:
      pd.DataFrame of features
    """
    assert "soil" in input_dfs
    soil_df = input_dfs["soil"]
    if "drainage_class" in soil_df.columns:
        soil_df["drainage_class"] = soil_df["drainage_class"].astype(str)
        # one hot encoding for categorical data
        soil_one_hot = pd.get_dummies(soil_df, prefix="drainage")
        soil_df = pd.concat([soil_df, soil_one_hot], axis=1).drop(
            columns=["drainage_class"]
        )
    soil_features = soil_df

    # Feature design for time series
    index_cols = [KEY_LOC, KEY_YEAR]
    period_length = "month"
    assert "meteo" in input_dfs
    weather_df = input_dfs["meteo"]
    weather_df = _add_period(weather_df, period_length)

    fpar_df = None
    if "fpar" in input_dfs:
        fpar_df = input_dfs["fpar"]
        fpar_df = _add_period(fpar_df, period_length)

    ndvi_df = None
    if "ndvi" in input_dfs:
        ndvi_df = input_dfs["ndvi"]
        ndvi_df = _add_period(ndvi_df, period_length)

    soil_moisture_df = None
    if "soil_moisture" in input_dfs:
        soil_moisture_df = input_dfs["soil_moisture"]
        soil_moisture_df = _add_period(soil_moisture_df, period_length)

    # cumulative sums
    weather_df = weather_df.sort_values(by=index_cols + ["date"])

    # Daily growing degree days
    # gdd_daily = max(0, tavg - tbase)
    # TODO: replace None in clip(0.0, None) with upper threshold.
    weather_df["tavg"] = weather_df["tavg"].astype(float)
    weather_df["gdd"] = (weather_df["tavg"] - GDD_BASE_TEMP[crop]).clip(
        0.0, GDD_UPPER_LIMIT[crop]
    )
    weather_df["cum_gdd"] = weather_df.groupby(index_cols, observed=True)[
        "gdd"
    ].cumsum()
    weather_df["cwb"] = weather_df["cwb"].astype(float)
    weather_df["prec"] = weather_df["prec"].astype(float)
    weather_df = weather_df.sort_values(by=index_cols + ["date"])
    weather_df["cum_cwb"] = weather_df.groupby(index_cols, observed=True)[
        "cwb"
    ].cumsum()
    weather_df["cum_prec"] = weather_df.groupby(index_cols, observed=True)[
        "prec"
    ].cumsum()

    ## C.J.: Added some new features here
    # Growing Degree Days (GDD)
    Tbase = 8 if crop == 'maize' else 5 # crop specific base temperatures
    weather_df['GDD'] = np.maximum(weather_df["tavg"] - Tbase, 0) # growing degree days: sum of all temperatures above base temperature
    weather_df['Cum_GDD'] = weather_df.groupby(index_cols, observed=True)['GDD'].cumsum()
    
    weather_df["rad"] = weather_df["rad"].astype(float)
    weather_df["cum_rad"] = weather_df.groupby(index_cols, observed=True)[
        "rad"
    ].cumsum()
    # radiation-based features
    weather_df["cum_PAR"] = 0.48 * weather_df["cum_rad"] # portion of PAR from tota radiation
    # temperature-based features
    Topt = 25 if crop=='maize' else 20; sigma = 7.0
    weather_df['Photo_T_Stress'] = np.exp(-((weather_df["tavg"] - Topt)**2) / (2 * sigma**2)) # phototsynthesis stress indicator
    weather_df['Water_Stress_Index'] = weather_df["cum_prec"] / (weather_df["cum_prec"] + 300.0) # water stress index
    weather_df['RUE_index'] = weather_df["cum_PAR"] * weather_df['Photo_T_Stress'] * weather_df['Water_Stress_Index'] # Radiation Use Efficiency (RUE) Index
    # Farquhar photosynthesis proxy
    weather_df['Photo_Farquhar_approx'] = compute_farquhar_proxy(weather_df["tavg"], weather_df["cum_prec"], weather_df["cum_rad"])
    ##


    if fpar_df is not None:
        fpar_df = fpar_df.sort_values(by=index_cols + ["date"])
        fpar_df["fpar"] = fpar_df["fpar"].astype(float)
        fpar_df["cum_fpar"] = fpar_df.groupby(index_cols, observed=True)[
            "fpar"
        ].cumsum()

    if ndvi_df is not None:
        ndvi_df = ndvi_df.sort_values(by=index_cols + ["date"])
        ndvi_df["ndvi"] = ndvi_df["ndvi"].astype(float)
        ndvi_df["cum_ndvi"] = ndvi_df.groupby(index_cols, observed=True)[
            "ndvi"
        ].cumsum()

    # Aggregate by period
    avg_weather_cols = ["tmin", "tmax", "tavg", "prec", "rad", "cum_cwb"]
    max_weather_cols = ["cum_gdd", "cum_prec", "Cum_GDD", "cum_rad"] # C.J.: added Cum_GDD and cum_rad
    avg_weather_aggrs = {ind: "mean" for ind in avg_weather_cols}
    max_weather_aggrs = {ind: "max" for ind in max_weather_cols}
    avg_ft_cols = {ind: "mean_" + ind for ind in avg_weather_cols}
    max_ft_cols = {ind: "max_" + ind for ind in max_weather_cols}

    # NOTE: combining max and avg aggregation
    weather_aggrs = {
        **avg_weather_aggrs,
        **max_weather_aggrs,
    }

    weather_fts = _aggregate_by_period(
        weather_df, index_cols, "period", weather_aggrs, {**avg_ft_cols, **max_ft_cols}
    )

    ## C.J.: Temperature Stress (HSD, CSD) and Water Stress (DD, WD)
    count_thresh_cols = {
        "tmin": ["<", "0"],  # Frost Days (<0 degrees)
        "tmin": ["<", "5"],  # Cold Stress Days (<5 degrees)
        "prec": ["<", "1"],  # Dry Days (<1mm precipitation); ToDo. Divide by total days for fraction
        "prec": [">", "10"],  # Wet Days (>10mm precipitation); ToDo. Divide by total days for fraction
    }
    # add Heat Stress Days (>35 degrees for maize, >30 degrees for wheat)
    count_thresh_cols["tmax"] = [">", "35"] if crop == 'maize' else [">", "30"]
    ## 
    
    #count_thresh_cols = {
    #    "tmin": ["<", "0"],  # degrees
    #    "tmax": [">", "35"],  # degrees
    #    "prec": ["<", "1"],  # mm (0 does not make sense, prec is always positive)
    #}
    operator_to_bool = {">": True, "<": False}
    # count time steps matching threshold conditions
    for ind, thresh in count_thresh_cols.items():
        threshold_exceed = operator_to_bool.get(thresh[0])
        # Assert to ensure that the operator is valid
        assert threshold_exceed is not None, f"Invalid operator {thresh[0]} for {ind}"
        threshold = float(thresh[1])
        if "_" in ind:
            ind = ind.split("_")[0]

        ft_name = ind + "".join(thresh)
        ind_fts = _count_threshold(
            weather_df,
            index_cols,
            "period",
            ind,
            threshold_exceed,
            threshold,
            ft_name,
        )

        weather_fts = weather_fts.merge(ind_fts, on=index_cols, how="left")
        weather_fts = weather_fts.fillna(0.0)

    all_fts = soil_features.merge(weather_fts, on=[KEY_LOC])
    if fpar_df is not None:
        fpar_fts = _aggregate_by_period(
            fpar_df,
            index_cols,
            "period",
            {"cum_fpar": "max"},
            {"cum_fpar": "max_cum_fpar"},
        )
        all_fts = all_fts.merge(fpar_fts, on=index_cols)

    if ndvi_df is not None:
        ndvi_fts = _aggregate_by_period(
            ndvi_df,
            index_cols,
            "period",
            {"cum_ndvi": "max"},
            {"cum_ndvi": "max_cum_ndvi"},
        )
        all_fts = all_fts.merge(ndvi_fts, on=index_cols)

    if soil_moisture_df is not None:
        soil_moisture_fts = _aggregate_by_period(
            soil_moisture_df, index_cols, "period", {"ssm": "mean"}, {"ssm": "mean_ssm"}
        )
        all_fts = all_fts.merge(soil_moisture_fts, on=index_cols)

    return all_fts
