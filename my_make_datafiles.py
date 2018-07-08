import sys
import os
import hashlib
import struct
import subprocess
import collections
import json
import tensorflow as tf
from tensorflow.core.example import example_pb2


dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506

VOCAB_SIZE = 200000
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data


def chunk_file(in_file, chunks_dir, set_name):
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1


def chunk_all(finished_files_dir):
  # Make a dir to hold the chunks
  chunks_dir = os.path.join(finished_files_dir, 'chunks')
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  for set_name in ['test']:
    in_file = os.path.join(finished_files_dir, '%s.bin' % set_name)
    print "Splitting %s data into chunks..." % set_name
    chunk_file(in_file, chunks_dir, set_name)
  print "Saved chunked data in %s" % chunks_dir


def tokenize_stories(asset_json_all, raw_stories_dir, tokenized_stories_dir):
  """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
  mapf = open('mapping.txt', 'w')
  for i, asset in enumerate(asset_json_all):
    print 'Processed: %d/%d'%(i, len(asset_json_all))
    for item in asset_json_all[asset]:
      item_file_name = '%s_%0.3f_%0.3f.txt'%(asset, item['start'], item['end'])
      item_file = os.path.join(raw_stories_dir, item_file_name)
      fid = open(item_file, 'w')
      item['fullText'] = [line.encode('utf-8').strip() for line in item['fullText']]
      fid.write(''.join(item['fullText']))
      fid.close()

      mapf.write('%s \t %s\n'%(item_file, os.path.join(tokenized_stories_dir, item_file_name)))
      
  mapf.close()

  command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
  subprocess.call(command)
  print "Stanford CoreNLP Tokenizer has finished."
  # os.remove("mapping.txt")

  # Check that the tokenized stories directory contains the same number of files as the original directory
  num_orig = len(os.listdir(raw_stories_dir))
  num_tokenized = len(os.listdir(tokenized_stories_dir))
  if num_orig != num_tokenized:
    raise Exception("The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (tokenized_stories_dir, num_tokenized, raw_stories_dir, num_orig))
  print "Successfully finished tokenizing %s to %s.\n" % (raw_stories_dir, tokenized_stories_dir)


def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines


def hashhex(s):
  """Returns a heximal formated SHA1 hash of the input string."""
  h = hashlib.sha1()
  h.update(s)
  return h.hexdigest()


def get_url_hashes(url_list):
  return [hashhex(url) for url in url_list]


def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."


def get_art_abs(story_file):
  lines = read_text_file(story_file)

  # Lowercase everything
  lines = [line.lower() for line in lines]

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  lines = [fix_missing_period(line) for line in lines]

  # Separate out article and abstract sentences
  article_lines = []
  highlights = []
  next_is_highlight = False
  for idx,line in enumerate(lines):
    if line == "":
      continue # empty line
    elif line.startswith("@highlight"):
      next_is_highlight = True
    elif next_is_highlight:
      highlights.append(line)
    else:
      article_lines.append(line)

  # Make article into a single string
  article = ' '.join(article_lines)

  # Make abstract into a signle string, putting <s> and </s> tags around the sentences
  abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])

  return article, abstract


def write_to_bin(tokenized_stories_dir, out_file, makevocab=False):
  """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
  tokenized_stories = os.listdir(tokenized_stories_dir)
  num_stories = len(tokenized_stories)

  # print "Making bin file for URLs listed in %s..." % url_file
  # url_list = read_text_file(url_file)
  # url_hashes = get_url_hashes(url_list)
  # story_fnames = [s+".story" for s in url_hashes]
  # num_stories = len(story_fnames)

  # if makevocab:
  #   vocab_counter = collections.Counter()

  with open(out_file, 'wb') as writer:
    for idx, story_file in enumerate(tokenized_stories):
      if idx % 1000 == 0:
        print "Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories))

      # Get the strings to write to .bin file
      story_file = os.path.join(tokenized_stories_dir, story_file)
      article, abstract = get_art_abs(story_file)

      # Write to tf.Example
      tf_example = example_pb2.Example()
      tf_example.features.feature['article'].bytes_list.value.extend([article])
      tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))

      # Write the vocab to file, if applicable
      if makevocab:
        art_tokens = article.split(' ')
        abs_tokens = abstract.split(' ')
        abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
        tokens = art_tokens + abs_tokens
        tokens = [t.strip() for t in tokens] # strip
        tokens = [t for t in tokens if t!=""] # remove empty
        vocab_counter.update(tokens)

  print "Finished writing file %s\n" % out_file

  # # write vocab to file
  # if makevocab:
  #   print "Writing vocab file..."
  #   with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
  #     for word, count in vocab_counter.most_common(VOCAB_SIZE):
  #       writer.write(word + ' ' + str(count) + '\n')
  #   print "Finished writing vocab file"


def check_num_stories(stories_dir, num_expected):
  num_stories = len(os.listdir(stories_dir))
  if num_stories != num_expected:
    raise Exception("stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))


if __name__ == '__main__':
  if len(sys.argv) != 3:
    print "USAGE: python my_make_datafiles.py <caption_dir> <asset_list>"
    sys.exit()
  root_dir = sys.argv[1]
  asset_list_file = sys.argv[2]

  asset_list_file = os.path.join(root_dir, asset_list_file)
  asset_list = []
  for line in open(asset_list_file):
    if line[0] == '#':
      continue
    asset_list.append(line.strip())
  print 'Found %d asset id'%len(asset_list)

  # asset_json_all = {}
  # for i, asset in enumerate(asset_list):
  #   asset_file = os.path.join(root_dir, 'viewers-choice-output', asset+'.json')
  #   try:
  #     asset_json = json.load(open(asset_file))
  #     asset_json = [item for item in asset_json if item['type']=='Program']
  #     asset_json_all[asset] = asset_json
  #   except:
  #     print 'Not found %d: %s'%(i, asset)

  # raw_stories_dir = os.path.join(root_dir, 'raw_stories')
  tokenized_stories_dir = os.path.join(root_dir, 'tokenize_stories')
  finished_files_dir = os.path.join(root_dir, 'finished_files')

  # Create some new directories
  # if not os.path.exists(raw_stories_dir): os.makedirs(raw_stories_dir)
  # if not os.path.exists(tokenized_stories_dir): os.makedirs(tokenized_stories_dir)
  # if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

  # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
  # tokenize_stories(asset_json_all, raw_stories_dir, tokenized_stories_dir)

  # Read the tokenized stories, do a little postprocessing then write to bin files
  # write_to_bin(tokenized_stories_dir, os.path.join(finished_files_dir, "test.bin"))

  # # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
  chunk_all(finished_files_dir)
