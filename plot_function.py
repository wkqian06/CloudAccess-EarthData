from osgeo import gdal
from rasterio.features import geometry_mask
import rasterio
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt

def ndarry_mask(lon, lat, data_ndarray, country_point=[-98.5, 39.8]):
    """
    Create a mask for the data based on contry borders
    """
    LON, LAT = np.meshgrid(lon, lat)
    countries = cfeature.NaturalEarthFeature(category='cultural', scale='110m', facecolor='none',
                                             name='admin_0_countries')

    # Extract the country geometry from countries feature
    us_geometry = None
    for country in countries.geometries():
        if country.contains(Point(country_point[0], country_point[1])):  # (lon, lat) in the country_point
            us_geometry = country  # get the geometry of the country
            break

    # Convert the US geometry to a GeoDataFrame
    country_gdf = gpd.GeoDataFrame(geometry=[us_geometry])

    # Define the transform - this relates pixel coordinates to geographical coordinates
    transform = rasterio.transform.from_bounds(np.min(LON), np.min(LAT), np.max(LON), np.max(LAT), LON.shape[1],
                                               LON.shape[0])

    # Create mask for continental US states
    mask = geometry_mask(country_gdf.geometry, transform=transform, invert=False,
                         out_shape=(LON.shape[0], LON.shape[1]))
    mask = np.flipud(mask)  # flip the mask to match the data

    data_mask = data_ndarray
    data_mask[mask] = np.nan

    return data_mask


def map_plot(lon, lat, data_ndarry, zoom_region=False, bounds=False, projection=ccrs.PlateCarree(), c_map=False,
             cb_extent='both'):
    LON, LAT = np.meshgrid(lon, lat)

    cmap = mpl.cm.jet

    if c_map == False:
        c_map = 'jet'

    fig = plt.figure(figsize=(16, 9))

    main_ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    if zoom_region != False:
        main_ax.set_extent(zoom_region)
        gl = main_ax.gridlines(draw_labels=True, color='gray', linestyle='--',
                               xlocs=np.arange(-180, 180 + 5, 5),
                               ylocs=np.arange(-90, 90 + 5, 5))
        main_ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='lightgray', facecolor='none')
    else:
        zoom_region = [-180,180,-90,90]
        gl = main_ax.gridlines(draw_labels=True, color='gray', linestyle='--')
    gl.top_labels, gl.right_labels = False, False

    
    main_ax.add_feature(cfeature.BORDERS, linewidth=0.75)
    main_ax.add_feature(cfeature.COASTLINE, linewidth=1)

    main_ax.grid()

    if bounds != False:
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend=cb_extent)
        cc = main_ax.contourf(LON, LAT, data_ndarry, extent=zoom_region, cmap=plt.get_cmap(c_map), norm=norm,
                              levels=bounds,
                              transform=projection, extend='both')

    else:
        cc = main_ax.contourf(LON, LAT, data_ndarry, extent=zoom_region, cmap=plt.get_cmap(c_map),
                              transform=projection, extend=cb_extent)
    cbar = fig.colorbar(cc, ax=main_ax, orientation='horizontal', fraction=.05, pad=0.05, extend=cb_extent)
    # cbar.set_label('mm/h', fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    return fig, main_ax, cbar
