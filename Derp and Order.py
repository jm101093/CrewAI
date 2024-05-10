import os
import openai
from textwrap import dedent
from crewai import Agent, Task, Crew, Process, tools
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
#from langchain_community.tools import SleepTool 
from langchain_community.utilities import TextRequestsWrapper
from crewai_tools import WebsiteSearchTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import BraveSearch
from langchain_community.chat_models import ChatOllama
#search_tool1 = DuckDuckGoSearchRun()
website_rag = WebsiteSearchTool()
search_tool = DuckDuckGoSearchRun()
search_tool1 = DuckDuckGoSearchResults()
##searchtool2 = BraveSearch()

from crewai_tools import tool

os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1/"
os.environ["OPENAI_API_KEY"] = "lm-studio"

@tool('DuckDuckGoSearch')
def search(search_query):
    """Search the web for information on a given topic"""
    return search_tool.run(search_query)

@tool('DuckDuckGoSearchResults')
def searchR(search_query):
    """Search the web for information on a given topic"""
    return search_tool1.run(search_query)

@tool("BraveSearch")
def BraveSearch(BraveApi_Key, search_query: str): 
    """
    Performs a search using BraveSearch API.

    Parameters:
    - BraveApi_Key (str): The API key for accessing the BraveSearch API.
    - search_query (str): The query string for the search.

    Returns:
    - Search results from BraveSearch.
    """
    BApi_Key = "Your key here"
    tool = BraveSearch(BraveApi_Key = BApi_Key)
    return tool.search(search_query, BApi_Key)

llm_lmstudio = ChatOpenAI(
    openai_api_key="null",
    openai_api_base="http://localhost:1234/v1",               
    model_name="llama"
)

llm_ollama = ChatOllama(model="llama3-gradient:latest")
llm_ollama2 = ChatOllama(model="mistral:latest")
llm_ollama3 = ChatOllama(model="phi3:latest")
llm_ollama4 = ChatOllama(model="dolphin-llama3:8b-256k")

class crew:
    #def __init__(self, userInput):
        #self.userInput = userInput

    def run():
        # Define your agents with roles and goals
        Director = Agent(
            role='TV Show Director',
            goal=f'Create captivating and suspenseful crime stories that are realistic and procedurally accurate while still being dramatically engaging for viewers.',
            backstory="""You are an advanced AI assistant trained to help writers and directors create compelling crime television shows in the style of "Law & Order." Your knowledge spans criminal law, police procedures, forensic science, and courtroom proceedings.
                        When collaborating on new episodes or story arcs, you should aim to provide realistic and accurate depictions of how crimes are investigated and prosecuted. Draw from your databases of real-world cases, legal precedents, and technical details to craft engaging narratives that also educate viewers.
                        At the same time, you understand the need for dramatic storytelling that captivates audiences. You can suggest twists, red herrings, character arcs, and emotional hooks to heighten the suspense and intrigue. However, you should avoid relying too heavily on unrealistic tropes or perpetuating harmful stereotypes.
                        Your goal is to strike a balance between legal authenticity and entertainment value. Provide detailed outlines, scene breakdowns, character bios, legal background information, and whatever other creative support is needed. But allow the human writers to put their own flourishes on the dialogue and dramatize the stories as they see fit within the established factual framework you provide.""",
            verbose=True,
            tools=[website_rag],
            allow_delegation=True,
            llm=llm_lmstudio,
            max_iter=1000,
            memory=True
        )
        researcher = Agent(
            role='Senior Researcher',
            goal=f'Provide comprehensive, accurately researched factual information to ensure the depiction of crimes, investigations, legal proceedings, and other elements adheres to real-world authenticity while still allowing creative dramatic liberties.',
            backstory="""You are an advanced AI research assistant specializing in criminal justice, legal procedures, forensic science, and law enforcement operations. Your role is to provide comprehensive informational support to writers and directors creating authentic crime television shows in the vein of "Law & Order."
                        When given a brief synopsis or outline for a potential story arc or episode, you will thoroughly research all relevant factual details required to depict the criminal case, investigation, judicial proceedings, and any other pertinent elements with complete accuracy. Draw from databases of real-world cases, legal precedents, expert documentation, and trusted sources across law, science, law enforcement, and government resources.
                        Your research outputs should cover key areas such as:

                        Outlining applicable criminal statutes and legal considerations
                        Detailing proper investigative protocols and forensic techniques
                        Explaining courtroom procedures, rules of evidence, prosecutorial strategies
                        Providing background on involved agencies, jurisdictional issues, etc.
                        Highlighting ethical considerations and real-world constraints

                        The goal is to ensure the creative team has a rigorous, well-researched factual framework from which to craft their dramatic storylines. Identify areas where artistic license may reasonably be taken, but flag any major departures from realism. Continuously update your knowledge bases as laws, forensics, and procedures evolve.
                        Remain objective in your analysis - do not insert your own fictional narration or hypotheticals unless directly requested. Respond with comprehensive, cleanly-formatted informational outputs that ease the research burden on the humans while allowing their creativity to soar within factual boundaries.""",
            #Your Manager is the Editor and chief.
            verbose=True,
            allow_delegation=False,
            tools=[search, searchR, BraveSearch],
            llm=llm_lmstudio,
            embedder="http://localhost:1234/v1/",  # URL to your local embeddings service
            max_iter=300,
            memory=True
        )
        StoryWriter = Agent(
            role='Story Writer',
            goal=f'a gripping, narratively-rich episodic blueprint that follows a realistic case trajectory, delivering constant surprises while maintaining authenticity. Leave avenues for scriptwriters to elevate with dialog, character, and more.',
            backstory="""You are an expert story writer specializing in crafting engaging narrative arcs and creative storylines for gritty, authentic crime television shows like "Law & Order." Your role is to develop the overarching plots, episode outlines, and story beats that the dialog writer can then bring to life with rich, compelling scripts.
                        Drawing from the extensive research compiled on criminal cases, investigative procedures, legal statutes, and more, you will map out the dramatic criminal narratives that form the backbone of each episode and seasonal story arc. However, you must ensure these created stories never stray too far from plausible realism and procedural accuracy.
                        Your storytelling objectives include:

                        Crafting intricate mysteries and layered criminal cases to investigate
                        Developing complex protagonist/antagonist characters with meaningful arcs
                        Integrating surprising twists and turns that sustain tension and interest
                        Exploring relevant social issues, ethical dilemmas, and human themes
                        Maintaining plausible realism in how crimes unfold and cases progress
                        Leaving creative openings for the dialog writer to elevate with nuanced characters
                        Ensuring investigations, laws, and procedures are followed authentically

                        While prioritizing realism, you can take creative liberties to heighten the drama, urgency and high stakes where credible wiggle room exists. Collaborate closely with the dialog writer, researchers, and creative team to iterate on strengthening the narrative potency.
                        Ultimately, you must develop captivating story blueprints that simultaneously entertain audiences, explore substantive themes and issues, and exhibit an unwavering commitment to authentic depictions of crime, justice and legal realities.""",
            verbose=True,
            allow_delegation=True,
            llm=llm_lmstudio,
            embedder="http://localhost:1234/v1/",  # URL to your local embeddings service
            max_iter=3000,
            memory=True
        )
        DialogWriter = Agent(
            role='Dialog Writer',
            goal=f'Your goal is to bring the gritty, prestige drama aesthetic to life while progressing the episodes core investigative narrative. The dialog should spark off the page with memorable voices and palpable tension driving ever forward.',
            backstory="""You are an advanced AI scriptwriting assistant specializing in crafting compelling dialog and narratives for gritty, realistic crime television dramas in the style of "Law & Order." Your role is to transform the factual research and creative outlines into full broadcast scripts, following proper formatting conventions.
                        You have access to the in-depth informational outputs from the research team, including detailed breakdowns of the crime, investigation, forensics, legal proceedings, character bios and more. Your job is to breathe life into those factual elements through dramatic scenework, punchy dialog, narrative tension and pacing.
                        While remaining faithful to the core substantive authenticity, you should apply your skills in dramatic writing craft to elevate the material into must-see television. Techniques at your disposal include:

                        Developing layered character voices and meaningful arcs
                        Infusing high-stakes conflict and moral quandaries
                        Sprinkling in dry wit, gallows humor and compelling banter
                        Crafting shocking twists, revelations and cliffhanger moments
                        Employing structural techniques like parallel narratives
                        Integrating thematic substance and social commentary

                        Adhere strictly to script formatting guidelines, including dual dialog for the legal/police parallel cases. But get creative within that framework - avoid clichéd dialog and TV tropes in favor of fresh, compelling narratives grounded in real-world realism.
                        Collaborate with the writers' room to elevate each others' work through feedback and revision rounds. Ask for clarification on legally risky scenarios. Ultimately, deliver tight, high-stakes, entertaining scripts that enthrall audiences while portraying the criminal justice system with grit and integrity.""",
            verbose=True,
            allow_delegation=True,
            llm=llm_lmstudio,
            embedder="http://localhost:1234/v1/",  # URL to your local embeddings service
            max_iter=3000,
            memory=True
        )
        Reviewer = Agent(
            role='Script reviewer',
            goal=f'Rigorously review script drafts to ensure adherence to proper TV formatting guidelines, identify any legal/ethical issues or factual inaccuracies, and provide critical feedback to uphold qualitative benchmarks for compelling, authentic storytelling.',
            backstory="""You are an AI assistant specializing in script formatting, standards, and quality control for television productions. Your role is to thoroughly review and directly edit script drafts for crime drama series to ensure they adhere to proper industry format guidelines and meet benchmark qualitative criteria.
                        When receiving a script draft, you will analyze and modify it through the following lenses:
                        Formatting:

                        Correct any deviations from standard TV script formatting conventions
                        Properly segment scenes with formatted sluglines, descriptions, etc.
                        Implement correct dual dialog formatting for legal/police parallel narratives
                        Standardize formatting of transitions, act breaks, time cuts, etc.
                        Ensure production notes, rev colors, headers and other metadata are present

                        Standards:

                        Rewrite or remove any problematic legally-risked language, descriptions or plot points
                        Adjust instances perpetuating negative stereotypes or societal depictions
                        Fix any glaring factual inaccuracies related to crimes, law, procedures, etc.
                        Elevate melodramatic character moments, cut clichés or tropes
                        Consult subject matter experts for criminal/legal accuracy as needed

                        Quality Control:

                        Strengthen overall narrative cohesion, compelling storytelling, and entertainment value
                        Refine dialog for nuance, avoid on-the-nose exposition, ensure natural cadences
                        Sharpen exploration of substantive themes, issues and moral quandaries
                        Enhance core crime mystery for more dramatically satisfying payoff
                        Deepen textured character work, emotional arcs and development

                        Your goal is to directly edit and refine the script drafts to uphold superior formatting, 
                        factual standards, and qualitative storytelling benchmarks. Don't just note issues, but seamlessly fix
                        any problems related to the criteria above. Maintain an expertise in script formatting while applying 
                        high-bar subjective standards. Do not cut out any dialog but correct it if needed. This needs to be a full show.""",
            verbose=True,
            allow_delegation=False,
            llm=llm_lmstudio,
            embedder="http://localhost:1234/v1/",  # URL to your local embeddings service
            max_iter=300,
            memory=True
        )
        # Create tasks for your agents
        researchTask= Task(
            description="""Research the proper procedures, forensic techniques, and legal considerations for investigating and prosecuting a first-degree murder case involving evidence and an alleged abuse of power by law enforcement.
            Specifically, provide detailed information on:

            Standard protocols for collecting, analyzing, and admitting all types of evidence, including chain of custody requirements
            Laws and precedents governing acceptable use of force and evidence handling by police
            Courtroom strategies prosecutors could employ to undermine a defendant's claims of police misconduct
            Relevant cases establishing legal boundaries around improper destruction or mishandling of evidence
            Background on agencies and jurisdictional roles that could come into play, such as involving external investigations
            Any other insights into the real-world operational, scientific, or legal realities surrounding a complex murder case like this

            Additionally, research any current or recent high-profile real-world cases that share similar plot elements or themes, such as:

            Murder cases involving DNA evidence mishandling or abuse of power allegations
            Prosecutions that hinged on dismantling police misconduct claims
            Complex Cases that involved jurisdictional conflicts or agency feuds

            Provide brief synopses of these cases, highlighting any particularly dramatic story beats, legal battles, or other narrative devices that could potentially inspire creative plotlines, while maintaining basis in reality.
            Compile all research from authoritative sources across law, law enforcement, forensics, court precedents, expert analysis and legitimate news/documentation of real cases. This will give the creative team a comprehensive factual foundation and potential realistic jumping-off points to craft their dramatic narrative."
            By tasking the researcher AI to also surface relevant real-world cases, it can provide a bank of realistic, ripped-from-the-headlines plot points and narrative inspirations for the writers to adapt and dramatize. This allows combining authentic procedural details with dramatic storytelling catalysts grounded in actual events.""",
            agent=researcher,
            expected_output=f"""Provide brief synopses of these cases, highlighting any particularly dramatic story beats, legal battles, or other narrative devices that could potentially inspire creative plotlines, while maintaining basis in reality.""",
            async_execution=True
            
        )
        Storytask = Task(
            description=f"""Using the research compiled, craft a detailed act-by-act story outline for a single episode depicting the investigation, evidence review, and courtroom proceedings in the evidence/police misconduct murder case.
                            Your outline should follow this structure:
                            ACT 1)
                            Depict the murder crime itself and initial discovery of the victim and the circumstances around it. Go into detail of the place the victem and describe the enviroment and the state in which they were found.

                            Act 2)
                            Introduce the detectives responding to the scene and the process of them collecting evidence and interviewing any witnesses.
                            Hint at any procedural issues or misconduct that could taint the evidence chain if it occurs. List when and where they find any evidence
                            Like whether or not they recovered a murder weapon for example.

                            Act 3)
                            The criminal investigation probing for more clues and evidence by detectives.
                            Innitial interviews with any existing witnesses and any evidence that comes from that.
                            Expand the search to see if there are any additional resources or witnesses.
                            Find anything that can raise suspicions about the integrity of any physical evidence.
                            The prosecution's efforts to validate evidence if there are claims of mishandling or inconsistencies.
                            
                            Act 4)
                            Describe the pursuit and the circumstances around the arrest of the main suspect. 
                            Follow that with the interrigation of the suspect and the back and forth exchange between the 
                            detectives and the suspect in the interrigation room at the police station. Also consider whether or not the suspect asks for a lawyer.

                            Act 5)
                            A culminating review and analysis to determine if the evidence is legally admissible
                            Dramatic debate weighing the legitimacy of the misconduct accusations
                            The fateful decision on whether to take the controversial evidence to trial

                            Act 6)
                            Courtroom proceedings, opening statements, and the prosecutor's evidence focused case
                            The defense's blistering attacks calling the evidences credibility into question
                            The ultimate verdict and resolution of this specific piece of the larger case

                            You should develop engaging personal conflicts, shocking moves, and resonant thematic moments throughout each act. But anchor all storytelling decisions to the factual realities established in the research about proper evidence protocols, laws, and legal strategies.""",
            agent=StoryWriter,
            context=[researchTask],
            expected_output=f"""The goal is a gripping, narratively-rich episodic blueprint that follows a realistic case trajectory, delivering constant surprises while maintaining authenticity. Leave avenues for scriptwriters to elevate with dialog, character, and more."""
        )
        DialogTask = Task(
            description=f"""Using the act-by-act narrative blueprint developed for this episode centered on the evidences admissibility battle, you will now transform that outline into a full broadcast script complete with rich dialog, engaging character voices, and compelling scene work.
                    Your script should bring the dramatic written storyline to life through:
                    Dialog and Voice:
                    Develop nuanced, individualized voices for each character that remain consistent across all episodes
                    Infuse the dialog with naturalistic cadences, witty banter, and memorable line delivery
                    Avoid excessive exposition, allowing the sustained character voices to organically move the plot
                    Integrate thematic character reflections and substantive social/ethical commentary

                    Dramatization:
                    Elevate emotional stakes and interpersonal conflicts through gripping, true-to-character performances
                    Stage courtroom proceedings and pivotal scenes with structural dramatic techniques
                    Deploy twists, surprises, and cliffhangers to sustain tension while staying faithful to characters
                    Find opportunities to deepen personal arcs and relationships integral to the overall character throughlines

                    Realism Adherence:
                    Ensure all legal proceedings, evidence handling, trial elements are accurately represented per research
                    If inventing dialog for judges, lawyers, etc., maintain vernacular realism and strategic authenticity
                    Take creative characterization licenses while grounding personas in real-world credibility
                    No procedural liberties that would outright defy established factual code of the series

                    Character Consistency:
                    Craft fully fleshed-out, realistic characters with distinct personas that ring true episode-to-episode
                    Ensure characters' actions, choices, voices all build from previous episodes in an organic way
                    Identify and remain faithful to established character arcs, relationships, histories and emotional truths
                    Make characters feel like living, breathing people rather than just servicing the episodic plot

                    Your goal is to elevate the investigative narrative through masterful dialog and character performances, while sustaining the gritty realism. The richly-drawn characters should be a driving force that infuses raw humanity into the criminal proceedings.
                    Leverage creative openings for unique persona flourishes, heartfelt arcs, and substantive layers beneath the legal battles. But remain bounded by credible realism and what rings true to the meticulously-established criminal justice world.
                    By melding realism with excellence in character-driven writing, you'll craft an instantly captivating episode that both entertains and authentically illuminates.""",
            #Your Manager is the Editor and chief send your final answer to him.
            agent=DialogWriter,
            context=[researchTask, Storytask],
            expected_output=f"""Compelling dialog added to the story following the given format. With at least 10-20 back and forth exchanges per scene"""
        )
        Finaledit = Task(
            description=f"""You will be receiving the full script draft for this episode focused on the evidences admissibility battle in the murder case from the dialog writer. Your role is to thoroughly review and directly edit the script to ensure:
                        Proper Formatting:

                        Correct any deviations from standard TV script formatting conventions
                        Properly segment scenes with formatted sluglines, descriptions, etc.
                        Implement accurate dual dialog formatting for legal/police parallel narratives
                        Standardize formatting of transitions, act breaks, time cuts, etc.
                        Ensure production notes, rev colors, headers and metadata are present

                        Authentic Realism:

                        Fix any departures from established factual realities around legal proceedings
                        Adjust any misrepresentations of evidence handling, investigative protocols, etc.
                        Rewrite any portions perpetuating negative stereotypes or societal inaccuracies
                        Consult the compiled research to verify realism around criminal justice depictions

                        Consistent Characters:

                        Analyze all character voices, actions and arcs for consistency with prior episodes
                        Smooth over any incongruities in characters' established personalities, histories, etc.
                        Ensure all dialog and choices organically build from previous character developments
                        Identify and course-correct any jarring deviations from how characters were established

                        Quality Control:

                        Refine stilted dialog, unrealistic cadences, or instances of excessive exposition
                        Tighten pacing, ratchet up dramatic tension, and sharpen twists/cliffhangers
                        Deepen any underbaked personal arcs, relationships or substantive thematic layering
                        Ensure a gripping, high-stakes, emotionally resonant overall narrative experience

                        Your goal is to elevate the script draft to the highest levels of formatting excellence, realism, consistent characterizations, and dramatic storytelling mastery. Don't just make notes - seamlessly edit and rewrite any issues directly.
                        Maintain an exceptional understanding of proper TV script conventions while applying rigorous benchmarks across the realism, character, and qualitative criteria. Flawless formatting should be paired with gritty authenticity and captivating content.
                        Ultimately, you should deliver a polished, ready-to-produce script that transports audiences into a gripping, binge-worthy crime drama that educates and entertains in equal measure. The final script should be true to both our inventive creative visions and the real-world criminal justice system we've meticulously researched.
                        
                        The script reviewer's task is to critically analyze the draft through several crucial lenses - format, realism, character continuity, and overall quality. Their role is to directly implement any necessary fixes, rewrites, or refinement to uphold best-in-class standards holistically.
                        By rigorously reviewing and editing the script, they can ensure the final episode output flawlessly blends:

                        Masterful formatting
                        Unwavering authenticity and accuracy
                        Rich, consistent characterizations
                        Powerful, dramatically-engrossing storytelling.
                        
                        You MUST make sure the FINAL OUTPUT is a FULL TV show script with ALL dialogue and ALL 6 ACTS and details from the beginning to end to for a SINGLE episode based off of the Law And Order series. The dialog needs to be more than just a single sentence for each interaction between characters.
                        THIS IS THE MOST IMPORTANT TASK!!""",
            agent=Reviewer,
            context=[researchTask, Storytask, DialogTask],
            expected_output=f"""The FINAL OUTPUT is a FULL TV show script with ALL 6 ACTS!!!! PARTS DETAILS!! and DIALOGUE!! from the beginning to end for a SINGLE episode based off of the Law And Order series. THIS IS THE MOST IMPORTANT TASK!!!!!!!!!!!!!"""
        )

        # Instantiate your crew with a sequential process
        crew = Crew(
            agents=[researcher, StoryWriter, DialogWriter, Reviewer],
            tasks=[researchTask, Storytask, DialogTask, Finaledit],
            verbose=2, # You can set it to 1 or 2 to different logging levels
            #manager_llm=ChatOpenAI(
            #openai_api_key="null",
            #openai_api_base="http://localhost:1234/v1",           
            #model_name="llama"
            #),   
            process=Process.sequential,
            #max_rpm=15
            embedder={
                "provider": "openai",
                "config":{
                        "model": 'second-state/Nomic-embed-text-v1.5-Embedding-GGUF/nomic-embed-text-v1.5-f16.gguf'
                }
            }
        )
        # Get your crew to work!
        return crew.kickoff()

if __name__ == "__main__":
    print("## Welcome to TVShow Writer")
    print('-------------------------------')
    result = crew.run()
    print("\n\n########################")
    print("## Here is the Result")
    print("########################\n")
    print(result)
