import streamlit as st
import numpy as np
import pandas as pd
import laspy
import geopandas as gpd
from scipy.interpolate import griddata
from scipy.ndimage import maximum_filter
from io import BytesIO

# --- メインの処理関数 (新ロジック) ---
def process_las_in_tiles(las_file_buffer, tile_size, buffer, resolution, ground_class, min_height, window_size):
    """
    LASファイルをタイル処理して樹頂点を検出する
    最終修正版：Numpy配列を直接操作して結合する
    """
    las_file_buffer.seek(0)
    
    with laspy.open(las_file_buffer) as f:
        header = f.header
        srs = f.header.parse_crs()
        
        xmin, ymin, zmin = header.x_min, header.y_min, header.z_min
        xmax, ymax, zmax = header.x_max, header.y_max, header.z_max
        
        all_treetops_dfs = []
        
        x_tiles = np.arange(xmin, xmax, tile_size)
        y_tiles = np.arange(ymin, ymax, tile_size)
        
        total_tiles = len(x_tiles) * len(y_tiles)
        if total_tiles == 0:
            st.error("タイルサイズがデータ範囲より大きいか、データ範囲が不正です。")
            return None, None
            
        progress_bar = st.progress(0, text="タイルの準備中...")

        for i, tx in enumerate(x_tiles):
            for j, ty in enumerate(y_tiles):
                tile_num = i * len(y_tiles) + j + 1
                progress_bar.progress(tile_num / total_tiles, text=f"タイル {tile_num}/{total_tiles} を処理中...")

                tile_xmin, tile_xmax = tx - buffer, tx + tile_size + buffer
                tile_ymin, tile_ymax = ty - buffer, ty + tile_size + buffer
                
                points_in_tile_chunks = []
                f.seek(0)
                for points_chunk in f.chunk_iterator(1_000_000):
                    mask = (
                        (points_chunk.x >= tile_xmin) & (points_chunk.x < tile_xmax) &
                        (points_chunk.y >= tile_ymin) & (points_chunk.y < tile_ymax)
                    )
                    if np.any(mask):
                        points_in_tile_chunks.append(points_chunk[mask])
                
                if not points_in_tile_chunks:
                    continue
                
                # --- ここからが根本的な修正点 ---
                # 1. 各チャンクから生のNumpy配列を抽出
                list_of_raw_arrays = [chunk.array for chunk in points_in_tile_chunks]
                # 2. 生のNumpy配列のリストを結合
                concatenated_raw_array = np.concatenate(list_of_raw_arrays)
                # 3. 結合した配列を再度laspyのPointRecordオブジェクトに変換し、座標のスケーリングを復元
                query_points = laspy.ScaleAwarePointRecord(
                    concatenated_raw_array, header.point_format, scales=header.scales, offsets=header.offsets
                )
                # --- 修正ここまで ---

                ground_points = query_points[query_points.classification == ground_class]
                if len(ground_points) < 10:
                    continue
                
                try:
                    grid_x, grid_y = np.mgrid[tx:tx+tile_size:resolution, ty:ty+tile_size:resolution]
                    
                    dtm = griddata((ground_points.x, ground_points.y), ground_points.z, (grid_x, grid_y), method='nearest')
                    dsm = griddata((query_points.x, query_points.y), query_points.z, (grid_x, grid_y), method='nearest')
                    
                    dchm = dsm - dtm
                    dchm[dchm < 0] = 0
                    
                    max_filtered = maximum_filter(dchm, size=window_size, mode='constant', cval=0)
                    treetops_mask = (dchm == max_filtered) & (dchm > min_height)
                    
                    rows, cols = np.where(treetops_mask)
                    if len(rows) > 0:
                        heights = dchm[rows, cols]
                        x_coords = grid_x[rows, cols]
                        y_coords = grid_y[rows, cols]
                        
                        tile_df = pd.DataFrame({
                            'x_coordinate': x_coords.ravel(),
                            'y_coordinate': y_coords.ravel(),
                            'height_m': heights.ravel()
                        })
                        all_treetops_dfs.append(tile_df)
                except Exception as e:
                    st.warning(f"タイル ({tx:.0f}, {ty:.0f}) の処理中にエラーが発生しました: {e}")

        progress_bar.empty()

        if not all_treetops_dfs:
            return None, None
            
        final_df = pd.concat(all_treetops_dfs, ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['x_coordinate', 'y_coordinate'])
        return final_df, srs

# --- Streamlit アプリのUI ---
st.set_page_config(page_title="大容量LAS対応 樹頂点検出", layout="wide")
st.title("🛰️ 大容量LAS対応 樹頂点検出アプリ (タイル処理)")

st.sidebar.header("⚙️ パラメータ設定")
st.sidebar.subheader("タイル処理設定")
tile_size_param = st.sidebar.number_input("タイルサイズ (m)", 50, 1000, 100, 50, help="一度に処理する領域の大きさ。PCのスペックが低い場合は小さくしてください。")
buffer_param = st.sidebar.number_input("バッファ (m)", 5, 50, 10, 5, help="タイルの境界での不整合を避けるための重複領域。")

st.sidebar.subheader("ラスター・樹頂点設定")
resolution_param = st.sidebar.number_input("ラスター解像度 (m)", 0.5, 5.0, 1.0, 0.5)
ground_class_param = st.sidebar.number_input("地面分類コード", 0, 255, 2, 1, help="ASPRS標準では `2` です。")
min_height_param = st.sidebar.slider("最低樹高 (m)", 1.0, 30.0, 5.0, 0.5)
window_size_param = st.sidebar.slider("探索ウィンドウサイズ (ピクセル)", 3, 21, 5, 2, help="奇数を指定してください。")

uploaded_file = st.file_uploader("LAS または LAZ ファイルをアップロード", type=['las', 'laz'])

if uploaded_file is not None:
    if st.button("樹頂点検出を実行", type="primary"):
        las_buffer = BytesIO(uploaded_file.getvalue())
        
        treetops_df, file_srs = process_las_in_tiles(
            las_buffer, tile_size_param, buffer_param, resolution_param, 
            ground_class_param, min_height_param, window_size_param
        )

        if treetops_df is not None and not treetops_df.empty:
            st.success(f"✅ **{len(treetops_df)}本**の樹頂点を検出しました。")
            
            st.subheader("📊 検出結果データ")
            st.dataframe(treetops_df.style.format({'height_m': '{:.2f}'}))

            gdf = gpd.GeoDataFrame(
                treetops_df,
                geometry=gpd.points_from_xy(treetops_df.x_coordinate, treetops_df.y_coordinate),
                crs=file_srs
            )
            gpkg_buffer = BytesIO()
            gdf.to_file(gpkg_buffer, driver='GPKG')

            st.download_button(
                label="GPKG形式でダウンロード",
                data=gpkg_buffer.getvalue(),
                file_name='treetops_tiled.gpkg',
                mime='application/geopackage+sqlite3'
            )

            st.subheader("🗺️ 樹頂点マップ")
            st.info("マップ表示は地理座標系（緯度経度）を想定しています。日本の平面直角座標系などのデータは正しく表示されない場合があります。")
            map_df = treetops_df.rename(columns={'x_coordinate': 'lon', 'y_coordinate': 'lat'})
            st.map(map_df)
            
        else:
            st.warning("指定された条件で樹頂点を検出できませんでした。パラメータを変更してお試しください。")
else:
    st.info("👆 上のエリアから解析したいLAS/LAZファイルをアップロードしてください。")