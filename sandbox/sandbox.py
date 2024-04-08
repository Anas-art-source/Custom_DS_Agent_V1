import sys
# Add the utils directory to the Python path
sys.path.insert(0, '/home/khudi/Desktop/my_own_agent/')
import pandas as pd
from code.xgboost_model import xgboost_model as xgboost_model_fit_predict
from tools.pants_data_api import pants_data_api
from tools.shirts_data_api import shirts_data_api
from tools.forecasting_model_inference import forecasting_model_inference_api
import io
import sys
import sys
import matplotlib.pyplot as plt
from io import StringIO, BytesIO


class PythonSandbox:
    def __init__(self):
        self.global_vars = globals()
        self.output = None

    def execute_code(self, code):
        # Dedent the code to adjust indentation
        dedented_code = code
        # Redirect stdout
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()

        # Execute the code
        try:
            exec(dedented_code,globals())
            # Check if a plot was created
            if plt.get_fignums():
                # Save the plot to a BytesIO object
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)

                #image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            self.output = redirected_output.getvalue()
                # Restore stdout
            sys.stdout = old_stdout

                #return {"output": output, "image_base64": image_base64}
            return {"result": 'SUCCESS', "output": self.output}

        except Exception as e:
            self.output = redirected_output.getvalue()
            # Restore stdout
            sys.stdout = old_stdout

            return {"result": 'FAILURE', "output":  f"Oops, there seems to be an error in your code. Make correction and retry.\n\nCode: {code}\n\nOutput:{self.output}\n\nError: {e}"}

