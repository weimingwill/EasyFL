from easyfl.tracking import service

parser = service.create_argument_parser()
args = parser.parse_args()

service.start_tracking_service(args.local_port)
