from .volleyball import *
from .nba import *
import pickle


TRAIN_SEQS_VOLLEY = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
VAL_SEQS_VOLLEY = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
TEST_SEQS_VOLLEY = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]


def read_dataset(args):
    if args.dataset == 'volleyball':
        data_path = args.data_path + args.dataset
        image_path = data_path + "/videos"

        train_data = volleyball_read_annotations(image_path, TRAIN_SEQS_VOLLEY + VAL_SEQS_VOLLEY, args.num_activities)
        train_frames = volleyball_all_frames(train_data)

        test_data = volleyball_read_annotations(image_path, TEST_SEQS_VOLLEY, args.num_activities)
        test_frames = volleyball_all_frames(test_data)

        train_set = VolleyballDataset(train_frames, train_data, image_path, args, is_training=True)
        test_set = VolleyballDataset(test_frames, test_data, image_path, args, is_training=False)

    elif args.dataset == 'volleyball_detect':
        data_path = args.data_path + "volleyball"
        image_path = data_path + "/videos"

        train_data = volleyball_read_annotations(image_path, TRAIN_SEQS_VOLLEY + VAL_SEQS_VOLLEY, args.num_activities)
        train_frames = volleyball_all_frames(train_data)

        test_data = volleyball_read_annotations(image_path, TEST_SEQS_VOLLEY, args.num_activities)
        test_frames = volleyball_all_frames(test_data)

        train_tracks = volleyball_read_tracks(data_path+"/volleyball_detections",train_data)
        test_tracks = volleyball_read_tracks(data_path+"/volleyball_detections",test_data)


        train_set = VolleyballDataset_detect(train_tracks,train_frames, train_data, image_path, args, is_training=True)
        test_set = VolleyballDataset_detect(test_tracks,test_frames, test_data, image_path, args, is_training=False)

    elif args.dataset == 'NBA':
        data_path = args.data_path + 'NBA/NBA_dataset'
        image_path = data_path + "/videos"

        # train_ids = [21801128,21801043,21801058,21801127,21801017,21801211,21801111,21801115, 21800980,21801213,21801110,21801220,21801141,21801107,21801086,21801051,21801155,21800995,21800991,21801217,21801095,21801218,21801150,21801121,21800960,21801172,21801039,21801054,21801171,21801015,21801124,21801144,21801098,21801013,21801019,21801008,21801004,21801106,21800934,21800999,21801149,21801224,21800972,21801108,21801072,21801161,21801065,21801090,21800979,21800919,21800994,21801053,21801114,21801151,21801216,21801074,21800938,21801168,21800909,21801012,21800929,21801135,21800982,21801160,21800949,21801228,21801097,21801226,21800952,21801140,21800983,21801052,21801061,21800997,21800975,21801126,21801189,21801064,21801125,21801209,21800987,21801089,21801163,21800992,21801219,21801136,21800976,21801068,21800966,21801139,21800981,21801011,21801085,21801116,21801119,21801214,21801048,21800989,21801113,21800985,21801158,21800964,21801215,21801046,21800984,21801091,21801164,21801002,21801056,21801175,21801167,21801069,21801087,21801060,21801100,21800973,21800971,21801112,21801018,21801157,21801003,21801006,21801067,21801001,21801147,21801094,21801057,21801230,21801131,21801154,21801042,21801156,21801225,21801045,21801104,21800990,21801210,21801014,21801145,21801148,21801099,21800996,21801223,21801088,21801134,21800968]
        # test_ids = [21800974,21801076,21801079,21800970,21800965,21801165,21801071,21801049,21801070,21800977,21801162,21801120,21801007,21801078,21801123,21801188,21800978,21801159,21801152,21801204,21801229,21800988,21801077,21801153,21801063,21801096,21801105,21801050,21801129]

        train_id_path = data_path + "/train_video_ids"
        test_id_path = data_path + "/test_video_ids"
        train_ids = read_ids(train_id_path)
        test_ids = read_ids(test_id_path)

        train_data = nba_read_annotations(image_path, train_ids)
        train_frames = nba_all_frames(train_data)

        test_data = nba_read_annotations(image_path, test_ids)
        test_frames = nba_all_frames(test_data)

        train_set = NBADataset(train_frames, train_data, image_path, args, is_training=True)
        test_set = NBADataset(test_frames, test_data, image_path, args, is_training=False)

    elif args.dataset == 'NBA_detect':
        data_path = args.data_path + 'NBA/NBA_dataset'
        image_path = data_path + "/videos"

        # train_ids = [21801128,21801043,21801058,21801127,21801017,21801211,21801111,21801115, 21800980,21801213,21801110,21801220,21801141,21801107,21801086,21801051,21801155,21800995,21800991,21801217,21801095,21801218,21801150,21801121,21800960,21801172,21801039,21801054,21801171,21801015,21801124,21801144,21801098,21801013,21801019,21801008,21801004,21801106,21800934,21800999,21801149,21801224,21800972,21801108,21801072,21801161,21801065,21801090,21800979,21800919,21800994,21801053,21801114,21801151,21801216,21801074,21800938,21801168,21800909,21801012,21800929,21801135,21800982,21801160,21800949,21801228,21801097,21801226,21800952,21801140,21800983,21801052,21801061,21800997,21800975,21801126,21801189,21801064,21801125,21801209,21800987,21801089,21801163,21800992,21801219,21801136,21800976,21801068,21800966,21801139,21800981,21801011,21801085,21801116,21801119,21801214,21801048,21800989,21801113,21800985,21801158,21800964,21801215,21801046,21800984,21801091,21801164,21801002,21801056,21801175,21801167,21801069,21801087,21801060,21801100,21800973,21800971,21801112,21801018,21801157,21801003,21801006,21801067,21801001,21801147,21801094,21801057,21801230,21801131,21801154,21801042,21801156,21801225,21801045,21801104,21800990,21801210,21801014,21801145,21801148,21801099,21800996,21801223,21801088,21801134,21800968]
        # test_ids = [21800974,21801076,21801079,21800970,21800965,21801165,21801071,21801049,21801070,21800977,21801162,21801120,21801007,21801078,21801123,21801188,21800978,21801159,21801152,21801204,21801229,21800988,21801077,21801153,21801063,21801096,21801105,21801050,21801129]

        train_id_path = data_path + "/train_video_ids"
        test_id_path = data_path + "/test_video_ids"
        train_ids = read_ids(train_id_path)
        test_ids = read_ids(test_id_path)

        train_data = nba_read_annotations(image_path, train_ids)
        train_frames = nba_all_frames(train_data)

        test_data = nba_read_annotations(image_path, test_ids)
        test_frames = nba_all_frames(test_data)

        all_tracks = pickle.load(open(data_path + '/normalized_detections.pkl', 'rb'))

        train_set = NBADataset_detect(all_tracks,train_frames, train_data, image_path, args, is_training=True)
        test_set = NBADataset_detect(all_tracks,test_frames, test_data, image_path, args, is_training=False)

    else:
        assert False

    print("%d train samples and %d test samples" % (len(train_frames), len(test_frames)))

    return train_set, test_set
