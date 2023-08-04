
# Import  modules
import sys
import argparse

from scipy.stats import norm
from distutils import util


def parse_ptf_stdin():


    description = 'pyPTF stdin parser'
#    example     = 'Example for single event mode:\n' + sys.argv[0] + ' --cfg ../cfg/ptf_main.config --event lefkada.json\n\n' + \
#                  'Example for rabbit connetion mode:\n'  + sys.argv[0] + ' --cfg ../cfg/ptf_main.config --mode rabbit'
    example     = 'EXAMPLES:\n=========\n' + \
                  'Example for single event mode:\n' + sys.argv[0] + ' --cfg cfg/ptf_main.config --event ../IO/earlyEst/2018_1025_zante_stat.json\n\n' #+ \
#                  'Example for rabbit connetion mode:\n'  + sys.argv[0] + ' --cfg cfg/ptf_main.config --mode rabbit' + '\n  '


    parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter, epilog = example)

    parser.add_argument('--seistype',              default = None,        help = 'BS or PS')
    parser.add_argument('--bathygrid',                 default = None,        help = 'File bathy')
    parser.add_argument('--hours',              default = None,        help = 'Nbr hour')
    parser.add_argument('--infile',            default = None,        help = 'file simu')
    parser.add_argument('--path',              default = None,        help = 'Main folder path')
    parser.add_argument('--cfg',               default = None,        help = 'Configuration file. Default: None')
    parser.add_argument('--mode',              default = 'event',     help = 'event: generate message for a single specific event (needs --event to be specified)' + \
                                                                             'rabbit: connect to the rabbit-mq and consume real-time events. Default = event')
    parser.add_argument('--event',             default = None,        help = 'seismic event parameter file. Default = None')
    parser.add_argument('--event_format',      default = 'json',      help = 'file format for event parameter file ([json]/xml/csv).')
    parser.add_argument('--rabbit_mode',       default = 'save',      help = 'rabbit-mq start sonsuming mode. save: hold and process the existing queue. clean: empty queue befor consuming. Default=save')
    parser.add_argument('--verbose',           default = False,       help = 'some verbose stdout. Default False')
    parser.add_argument('--preload',           default = 'No',        help = 'preload from tsumaps [Yes]/No')
    parser.add_argument('--preload_curves',    default = 'No',        help = 'preload hazard curves (gl/af/os) Yes/[No]')
    parser.add_argument('--preload_BS4',       default = 'No',        help = 'preload preload_BS4 only Yes/[No]')
    parser.add_argument('--preload_PS',        default = 'No',        help = 'preload preload_PS only Yes/[No]')
    parser.add_argument('--preload_scenarios', default = 'No',        help = 'preload PS and BS scenarios only Yes/[No]')
    parser.add_argument('--preload_mesh',      default = 'No',        help = 'preload slab meshes Yes/[No]')
    parser.add_argument('--preload_weight',    default = 'No',        help = 'preload wheigths Yes/[No]')
    parser.add_argument('--preload_discretization',      default = 'No',        help = 'preload discretizations Yes/[No]')
    parser.add_argument('--regions',           default = '-1',        help = 'regions to load [1-100]. -1=all. May select one ore more. Default=-1')
    parser.add_argument('--ignore_regions',    default = None,        help = 'regions to ignore [1-100]. Default=None')
    parser.add_argument('--pois',              default = '-1',        help = 'pois to load: -1=mediterranean:all pois in the mediterranean ' + \
                                                                             '; mediterranean-4: 1 POI every 4 in the Mediterranean Sea ' + \
                                                                             '; med09159,med09174: specific selected pois. Default=-1')
    parser.add_argument('--geocode_area',      default = 'No',        help = 'Get the area name from geocode search instead of json file. Can take 1.5 seconds. [No]/Yes.')
    parser.add_argument('--mag_sigma_fix',     default = 'No',        help = 'Fix the magnitude event sigma. If Yes take the mag_sigma_val value. [No]/yes')
    parser.add_argument('--mag_sigma_val',     default = '0.15',      help = 'Assumed magnitude event sigma. Needs --mag_sigma_fix=Yes. Default=0.15')
    parser.add_argument('--points_2d_ellipse', default = None,        help = 'Number of points to set the 2d ellipse. Default is set by the configuration file. ' + \
                                                                             'This option when used override the configuration value')
    parser.add_argument('--sigma_inn',         default = None,        help = 'real numbers indicating the proportion of standard deviation surrounded by each ellipse ' +\
                                                                             'to select all positions of interest. Default is set by the configuration file. ' + \
                                                                             'This option when used override the configuration value')
    parser.add_argument('--sigma_out',         default = None,        help = 'real numbers indicating the proportion of standard deviation surrounded by each ellipse ' +\
                                                                             'to select all positions of interest using a larger area to avoid bias. Default is set by the configuration file. ' + \
                                                                             'This option when used override the configuration value')
    parser.add_argument('--sigma',             default = None,        help = 'real numbers indicating the proportion of standard deviation surrounded by each ellipse. ' +\
                                                                             'Default is set by the configuration file. This option when used override the configuration value. ' + \
                                                                             'Updating this value will also update the \"negligible_probability\" value')
    parser.add_argument('--intensity_measure', default = 'gl',        help = 'tsunami Intesity Measure Method: gl, af, os. Default: gl')
    parser.add_argument('--compute_runUp',     default = 'n',         help = 'Compute Maximum run-up. [Yes]/No')
    parser.add_argument('--ps_type',           default = '1',         help = 'Ps probability type: 1,2. Default 1')
    parser.add_argument('--save_main_path',    default = None,        help = 'Save main Path. This option override configuration file. Default See Config file')
    parser.add_argument('--save_sub_path',     default = None,        help = 'Save sub Path. This option override configuration file. Default See Config file')
    parser.add_argument('--save_format',       default = None,        help = 'Format save file (npy/hdf5). This option override configuration file. Default See Config file')
    parser.add_argument('--alert_type',        default = 'best',      help = 'Type of alert level estimator to use [best/average/probabilityXX]. XX probability ranges: [05-50] Default: best')
    parser.add_argument('--production',        default = False,       help = 'Production of develop mode (for senting mail and rabbit messages). [False]/True',type=lambda x: bool(util.strtobool(x)), choices=[True, False])
    parser.add_argument('--rabbit_family',     default = 'neam',      help = 'Publishing Routing Key Family: neam, dpc, dpc_test, comtest. Default: neam')
    parser.add_argument('--pub_email',         default = False,        help = 'Send alert messages via email. [True]/False',type=lambda x: bool(util.strtobool(x)), choices=[True, False])
    parser.add_argument('--pub_rabbit',        default = False,       help = 'Send alert messages on rabbit. [False]/True',type=lambda x: bool(util.strtobool(x)), choices=[True, False])
    
    #new parameters for the whole workflow
    parser.add_argument('--kagan_weight',         default = 0,        help = 'Include kga in the evaluation process')
    parser.add_argument('--mare_weight',         default = 0,        help = 'Include tsu in the evaluation process')
    parser.add_argument('--group_sims',         default = 0,        help = 'Number of finished simulations in for the partial evaluation')
    parser.add_argument('--data_path', default = None, help = "Path where required data is provided")
    parser.add_argument('--run_path', default = None, help = "Path where workflow is executed output is stored")
    parser.add_argument('--templates_path', default = None, help = 'Path where configuration templates are stored')
    parser.add_argument('--parameters_file', default = None, help = 'Parameter_file')
    
    parser.add_argument('--event_id', default = 0, help = 'Event id') 
    
     
    # load arguments
    args = parser.parse_args()

    # if any
    if not sys.argv[1:]:
           print ("Use -h or --help option for Help")
           sys.exit(0)

    # first check on agrument consistency
    args = check_arguments(args=args)

    return args

def update_cfg(**kwargs):

    args   = kwargs.get('args', None)
    Config = kwargs.get('cfg', None)

    if(args.points_2d_ellipse != None):
       Config['Settings']['nr_points_2d_ellipse'] = args.points_2d_ellipse

    if(args.sigma_inn != None):
       Config['Settings']['nSigma_inn'] = args.sigma

    if(args.sigma_out != None):
       Config['Settings']['nSigma_out'] = ("%.9f" % (float(args.sigma)+0.5))

    if(args.sigma != None):
       Config['Settings']['nSigma'] = args.sigma
       negligible_probability = ("%.9f" % (2*norm.cdf(-1. * float(args.sigma))))
       Config['Settings']['negligible_probability'] = negligible_probability

    if(args.preload == 'Yes' or args.preload == 'yes' or args.preload == 'y'):
       args.preload_mesh = 'Yes'

    return Config

def check_arguments(**kwargs):

    args = kwargs.get('args', None)

    if(args.cfg == None):
        print ("Please provide a configuration file")
        sys.exit()


    # File format check
    if(args.event_format == 'json' or args.event_format == 'jsn'):
        args.event_format = 'jsn'
    elif(args.event_format == 'xml' or args.event_format == 'XML' or args.event_format == 'csv'):
        print(args.event_format + " event file format not jey supported. Exit")
        sys.exit()
    else:
        print(args.event_format + " event file format not recognized. Exit")
        sys.exit()

    # mode and event check. if mode == event, ann event file MUST be provided
    if(args.mode == 'event' and args.event == None):
        print('Please provide an event file (use --event)')
        sys.exit()


    args.compute_runUp = args.compute_runUp[0:1].lower()

    # check if event file exists
    #if (args.event != None and os.path.exists(args.event) == False):
    #    print('event file ' + args.event + ' not found. Exit')
    #    sys.exit()

    return args
