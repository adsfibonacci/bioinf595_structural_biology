import io

import logging

import param
import panel as pn
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
import holoviews as hv
from holoviews.operation.datashader import datashade, rasterize, shade, dynspread
from holoviews.operation import decimate
decimate.max_samples = 5000

from datashader.colors import viridis
from datashader.colors import colormap_select
from datashader.transfer_functions import set_background

import colorcet as cc

import sys
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    force=True
)

class MoleculePanel(pn.reactive.ReactiveHTML):
    smiles_list = param.List()
    label_list = param.List()
    _extension_name = 'molpanel'
    __javascript__ = ['https://unpkg.com/smiles-drawer@1.0.10/dist/smiles-drawer.min.js']
    _template = """<div id="molgrid" style="
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    "></div>"""
    _scripts = {
        'smiles_list': """
          while (molgrid.firstChild) {
            molgrid.removeChild(molgrid.firstChild);
          }
          let smilesDrawer = new SmilesDrawer.Drawer({
            width : 100,
            height : 100});

          for (let mol_index = 0; mol_index < data.smiles_list.length; mol_index++) {
            // Outer container
            let container = document.createElement('div');
            container.style.display = "flex";
            container.style.flexDirection = "column";
            container.style.alignItems = "center";
            container.style.width = "110px";

            // Canvas
            let canvas = document.createElement('canvas');
            canvas.style.border = "1px solid";
            canvas.width = 100;
            canvas.height = 100;

            // Label
            let labelDiv = document.createElement('div');
            labelDiv.style.marginTop = "4px";
            labelDiv.style.fontSize = "12px";
            labelDiv.style.textAlign = "center";
            labelDiv.style.wordWrap = "break-word";

            let labelText = "";
            if (data.label_list && data.label_list.length > mol_index) {
                labelText = data.label_list[mol_index];
            }
            labelDiv.textContent = labelText;

            container.appendChild(canvas);
            container.appendChild(labelDiv);
            molgrid.appendChild(container);

            SmilesDrawer.parse(data.smiles_list[mol_index], function(tree) {
                smilesDrawer.draw(tree, canvas, "light", false);
            });
          }
        """
    }

hv.extension("bokeh")
pn.extension('molpanel')


class DataExplorer:

    def __init__(self):
        self.df = None

        self.file_input = pn.widgets.FileInput(accept=".csv")
        self.x_select = pn.widgets.Select(name="X column")
        self.y_select = pn.widgets.Select(name="Y column")
        self.label_select = pn.widgets.Select(name="Label column")        
        self.smiles_select = pn.widgets.Select(name="SMILES column")

        self.max_instances = 5
        
        self.file_input.param.watch(self.load_data, "value")

        # Overlay the region of interest edit box (roi_box) layer over the provided plot
        # and then store the coordinates in the roi_stream
        self.roi_box = hv.Polygons([]).opts(
            hv.opts.Polygons(
                fill_alpha=0.2,
                line_color='white'))
        self.roi_stream = hv.streams.BoxEdit(source=self.roi_box)

        # this is watching the 'data' parameter of the roi_stream
        # and when it changes, it calls the callback function
        self.roi_stream.param.watch(self.callback, 'data')
        
      
        # create MoleculePanel that depicts molecules using javascript
        # and then update with a callback which molecules are shown
        self.mol_panel = MoleculePanel(smiles_list = [])


        
        self.layout = pn.Column("""
## Interactive Chemical Embedding
**Usage**:
Upload a .csv file with columns ['UMAP_1', 'UMAP_2', 'label', 'smiles']

**To select compounds**:

 1. On the toolbar, click the Box Edit Tool (rectangle with a plus in lower right corner)
 2. Double click to start selecting and then double click to stop selecting
 3. To move a selection click and drag it around
 4. To delete a selection, click a selection to activate it, and then use the DELETE key
""",
            self.file_input,
            pn.Row(
                self.x_select,
                self.y_select,
                self.label_select,
                self.smiles_select),
            pn.bind(
                self.plot,
                self.x_select,
                self.y_select,
                self.label_select,
                self.smiles_select))

    def load_data(self, event):
        if event.new is None:
            return

        bytes_io = io.BytesIO(event.new)
        self.df = hv.Dataset(pd.read_csv(bytes_io))

        cols = list(self.df.dimensions())
        self.x_select.options = cols
        self.y_select.options = cols
        self.label_select.options = cols
        self.smiles_select.options = cols
        
        if len(cols) >= 4:
            # these are currently hard coded
            self.x_select.value = cols[0]
            self.y_select.value = cols[1]
            self.label_select.value = cols[2]
            self.smiles_select.value = cols[3]

    def view_UMAP(self, x_field, y_field):
        points = hv.Points(
            self.df,
            kdims = [x_field, y_field],
            label="UMAP Embedding of Compounds").opts(
                width=800, height=800)
        map = rasterize(points)
        colormap = colormap_select(viridis)
        map = shade(map, cmap=colormap)
        map = dynspread(map, threshold=0.5)
        map = map.options(bgcolor='black')
        hover_points = decimate(points)
        hover_points.opts(tools=['hover'], alpha=0)
        return (map * hover_points)

    def plot(self, x_field, y_field, label_field, smiles_field):

        if self.df is None:
            return pn.pane.Markdown("Upload a CSV file.")

        if (x_field is None) or (y_field is None) or (label_field is None) or (smiles_field is None):
            return pn.pane.Markdown("Select columns.")

        if not is_numeric_dtype(self.df[x_field].dtype):
            return pn.pane.Markdown(f"X column must be numeric")
        if not is_numeric_dtype(self.df[y_field].dtype):
            return pn.pane.Markdown(f"Y column must be numeric")
        if x_field == y_field:
            return pn.pane.Markdown("X and Y must be different columns.")
        if not is_string_dtype(self.df[smiles_field].dtype):
            return pn.pane.Markdown(f"Smiles column must be 'str'")

        self.current_x_field = x_field
        self.current_y_field = y_field
        self.current_label_field = label_field
        self.current_smiles_field = smiles_field
        
        umap = self.view_UMAP(x_field, y_field)
        return pn.Row(umap * self.roi_box, self.mol_panel)
        
      
    def callback(self, event):
        data = event.new
        if not data or len(data['x0']) == 0:
            smiles_list = self.df[self.current_smiles_field].tolist()
            label_list = self.df[self.current_label_field].tolist()            
        else:
            selection = self.df.select(**{
                self.current_x_field.name : (data['x0'][0], data['x1'][0]),
                self.current_y_field.name : (data['y0'][0], data['y1'][0])})
            smiles_list = selection[self.current_smiles_field].tolist()
            label_list = selection[self.current_label_field].tolist()
  
        if not isinstance(smiles_list, list): smiles_list = [smiles_list]
        if not isinstance(label_list, list): label_list = [label_list]
        if len(smiles_list) > self.max_instances:
            smiles_list = smiles_list[0:5]
            label_list = label_list[0:5]
        self.mol_panel.smiles_list = smiles_list
        self.mol_panel.label_list = label_list

app = DataExplorer()
app.layout.servable()
