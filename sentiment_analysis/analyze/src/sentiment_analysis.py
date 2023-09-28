import argparse
from pathlib import Path
from typing import List

from parsoda import SocialDataApp
from parsoda.function.crawling.distributed_file_crawler import DistributedFileCrawler
from parsoda.function.crawling.parsing.flickr_parser import FlickrParser
from parsoda.function.crawling.parsing.parsoda_parser import ParsodaParser
from parsoda.function.crawling.parsing.twitter_parser import TwitterParser
from parsoda.model.driver.parsoda_pycompss_driver import ParsodaPyCompssDriver

from parsoda.function.filtering import HasEmoji, ContainsKeywords
from parsoda.function.mapping.classify_by_emoji import ClassifyByEmoji
from parsoda.function.reduction.reduce_by_emoji_polarity import ReduceByEmojiPolarity
from parsoda.function.analysis.two_factions_polarization import TwoFactionsPolarization
from parsoda.function.visualization.print_emoji_polarization import PrintEmojiPolarization
from parsoda.model.driver.parsoda_driver import ParsodaDriver
from parsoda.model.function.crawler import Crawler

def parsoda_sentiment_analysis(
    driver: ParsodaDriver,
    crawlers: List[Crawler],
    *, 
    num_partitions=-1, 
    chunk_size=64,
    emoji_file="./resources/input/emoji.json", 
    visualization_file="./resources/output/emoji_polarization.txt",
    keywords: str = "",
    keywords_separator: str = " ",
    keywords_threshold: int = 1
):  
    app = SocialDataApp("Sentiment Analysis", driver, num_partitions=num_partitions, chunk_size=chunk_size)
    app.set_crawlers(crawlers)
    app.set_filters([
        ContainsKeywords(
            keywords=keywords, 
            separator=keywords_separator, 
            threshold=keywords_threshold
        ),
        HasEmoji()
    ])
    app.set_mapper(ClassifyByEmoji(emoji_file))
    app.set_reducer(ReduceByEmojiPolarity())
    app.set_analyzer(TwoFactionsPolarization())
    app.set_visualizer(PrintEmojiPolarization(visualization_file))
    return app

if __name__ == "__main__":
    def type_input_format(arg: str):
        if arg=="twitter":
            return TwitterParser()
        elif arg=="flickr":
            return FlickrParser()
        elif arg=="parsoda":
            return ParsodaParser()
        else:
            raise Exception("Unsupported parser")
    
    parser = argparse.ArgumentParser(description='Sentiment Analysis application on top of PyCOMPSs')
    parser.add_argument(
        "input_dataset",
        type=Path,
        help="path to the input dataset"
    )
    parser.add_argument(
        "input_format",
        type=type_input_format,
        help="format of the input JSON dataset, can be: 'twitter', 'flickr', or 'parsoda'."
    )
    parser.add_argument(
        "emoji_file",
        type=Path,
        help="path to the emoji dataset"
    )
    parser.add_argument(
        "visualization_file",
        type=Path,
        help="path to the output file"
    )
    parser.add_argument(
        "--keywords", "-kw",
        type=str,
        default="",
        help="specifies the keywords of the topic for filtering items. Defaults to \"\" (empty string, filtering by keywords is disabled)."
    )
    parser.add_argument(
        "--keywords-separator", "-ksep",
        type=str,
        default=" ",
        help="specifies the separator of the specified keywords. Defaults to ' ' (space)."
    )
    parser.add_argument(
        "--keywords-threshold", "-kth",
        type=int,
        default=1,
        help="specifies the number of different keywords that must be contained in a filter item. Defaults to 1."
    )
    parser.add_argument(
        "--partitions", "-p",
        type=int,
        default=-1,
        help="specifies the number of data partitions."
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=128,
        help="specifies the size of data partitions in megabytes."
    )
    args = parser.parse_args()
    
    app = parsoda_sentiment_analysis(
        driver=ParsodaPyCompssDriver(),
        crawlers = [DistributedFileCrawler(args.input_dataset, args.input_format)],
        num_partitions=args.partitions,
        chunk_size=args.chunk_size,
        emoji_file=args.emoji_file,
        visualization_file=args.visualization_file,
        keywords=args.keywords,
        keywords_separator=args.keywords_separator,
        keywords_threshold=args.keywords_threshold
    )
    
    app.execute()