import py3dtiles.tileset as tileset
from pathlib import Path
import inspect
m = tileset.TileSet.from_file(Path(r"E:\2021-09-web三维\分层分户矢量文件\tile-5\2-2\tileset.json"))

m.set_properties_from_dict({'cccccccc':'kkkkkkkkkk'})
for c in m.get_all_tile_contents():
    table = c.body.batch_table
    print(inspect.getmembers( table.header))
    print(c)