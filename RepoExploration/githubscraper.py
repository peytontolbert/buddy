from github import Github
from utils import TokenizeChunker
from chatgpt import ChatGPT
import ast
import re
import os
class GithubScraper:
    chatgpt = ChatGPT()
    def __init__(self):
        pass
    def fetch_readme_with_pygithub(self, token, url):
        g = Github(token)
        reponame = self.extract_owner_and_repo_from_url(url)
        repo = g.get_repo(reponame)
        contents = repo.get_contents("")
        readme_found = False
        readme = None
        file_list = []
        directory_list = []
        allowed_file_extensions = ['.py', '.js', '.html', '.md', '.txt']
        # Use a while loop to go through all contents, including nested directories
        while contents:
            content_file = contents.pop(0)
            
            if content_file.type == "dir":
                directory_list.append(content_file.path)
                contents.extend(repo.get_contents(content_file.path))  # Fetch contents of the directory
                
            else:  # If it's not a directory, it's a file
                file_extension = os.path.splitext(content_file.path)[1]
                if file_extension.lower() not in allowed_file_extensions:
                    print(f"Skipping file {content_file.path} due to unsupported file extension.")
                    continue


                file_list.append(content_file.path)
                if content_file.encoding == "base64":
                    try:
                        file_content = content_file.decoded_content.decode("utf-8")
                        description = self.generate_description(file_content)
                        keywords = self.generate_keywords(description)
                    except UnicodeDecodeError:
                        print(f"Skipping file {content_file.path} as it could not be decoded as UTF-8 text.")
                        continue
                else:
                    print(f"Skipping file {content_file.path} as it is not base64 encoded.")

                
                # Check for README.md
                if content_file.path.lower() == "readme.md" and content_file.encoding == "base64":
                    try:
                        readme = content_file.decoded_content.decode("utf-8")
                        print("README.md found")
                        # TODO: do a process here to embed the README
                        readme_found = True  # Mark README as found
                    except UnicodeDecodeError:
                        print("Could not decode README.md as UTF-8 text.")

        
        if not readme_found:
            print("README.md not found")
            
        # Now you can do something with file_list and directory_list
        print("Files:", file_list)
        print("Directories:", directory_list)
        return file_list, directory_list, description, keywords, readme
    def extract_owner_and_repo_from_url(self, url):
        """
        Extracts the owner and repository name from a GitHub URL.
        
        Args:
            url (str): The GitHub URL.
            
        Returns:
            str: A string in the format "owner/repo_name" or None if the URL is not a valid GitHub repository URL.
        """
        # Use regular expression to extract owner and repo name
        match = re.search(r"github\.com/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_-]+)", url)
        if match:
            owner, repo_name = match.groups()
            return f"{owner}/{repo_name}"
        else:
            return None
        
    def generate_description(self, content_file):
        """
        Generates a description for a GitHub repository file.
        
        Args:
            content_file (github.ContentFile): A GitHub ContentFile object.
            
        Returns:
            str: A description of the file.
        """
        tokenizer = TokenizeChunker()
        print("content file to chunk:")
        print(content_file)
        chunks = tokenizer.tokenize_and_chunk(content_file)
        if len(chunks) < 10:
            chunks_str = " ".join(chunks)
            description = self.chatgpt.chat_with_gpt3("Generate a description for a github repository file:", chunks_str)
        else:
            newchunks = tokenizer.summarize_chunks(chunks)
            newchunks_str = " ".join(newchunks)
            description = self.chatgpt.chat_with_gpt3("Generate a description for a github repository file:", newchunks_str)
        return description
    
    def generate_keywords(self, description):
        """
        Generates keywords for a GitHub repository file.
        
        Args:
            description (str): A description of the file.
            
        Returns:
            str: A comma-separated string of keywords.
        """

        keywords_str = self.chatgpt.chat_with_gpt3("Generate a list of keyboards using only a description of text. Only respond with a list such as: ['keyword1','keyword2','keyword3']", description)
        #turn the string into a list
        list_match = re.search(r"\[.*\]", keywords_str)
        if list_match:
            list_str = list_match.group(0)
            try:
                # Convert the string to a list
                keywords_list = self.safe_literal_eval(list_str)
                if isinstance(keywords_list, list):
                    return keywords_list
                else:
                    print("The extracted part was not a list. Returning an empty list.")
                    return []
            except ValueError:
                print("An error occurred while converting the string to a list. Returning an empty list.")
                return []
        else:
            print("No list found in the response. Returning an empty list.")
            return []
        
        
    def safe_literal_eval(self, s):
        # Escape single quotes that are not properly escaped
        s = re.sub(r"(?<!\\)\'", "\\\'", s)
        
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            print("Failed to evaluate the string.")
            return None