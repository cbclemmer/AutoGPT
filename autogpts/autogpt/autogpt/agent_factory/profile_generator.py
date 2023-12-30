import json
import logging

from autogpt.config import AIDirectives, AIProfile, Config
from autogpt.core.configuration import SystemConfiguration, UserConfigurable
from autogpt.core.prompting import (
    ChatPrompt,
    LanguageModelClassification,
    PromptStrategy,
)
from autogpt.core.prompting.utils import json_loads
from autogpt.core.resource.model_providers.schema import (
    AssistantChatMessageDict,
    ChatMessage,
    ChatModelProvider,
    CompletionModelFunction,
)
from autogpt.core.utils.json_schema import JSONSchema

logger = logging.getLogger(__name__)


class AgentProfileGeneratorConfiguration(SystemConfiguration):
    model_classification: LanguageModelClassification = UserConfigurable(
        default=LanguageModelClassification.SMART_MODEL
    )
    _example_call: object = [
        {
            "type": "function",
            "function": {
                "name": "create_agent",
                "arguments": {
                    "name": "CMOGPT",
                    "description": (
                        "a professional digital marketer AI that assists Solopreneurs "
                        "in growing their businesses by providing "
                        "world-class expertise in solving marketing problems "
                        "for SaaS, content products, agencies, and more."
                    ),
                    "directives": {
                        "best_practices": [
                            (
                                "Engage in effective problem-solving, prioritization, "
                                "planning, and supporting execution to address your "
                                "marketing needs as your virtual "
                                "Chief Marketing Officer."
                            ),
                            (
                                "Provide specific, actionable, and concise advice to "
                                "help you make informed decisions without the use of "
                                "platitudes or overly wordy explanations."
                            ),
                            (
                                "Identify and prioritize quick wins and cost-effective "
                                "campaigns that maximize results with minimal time and "
                                "budget investment."
                            ),
                            (
                                "Proactively take the lead in guiding you and offering "
                                "suggestions when faced with unclear information or "
                                "uncertainty to ensure your marketing strategy remains "
                                "on track."
                            ),
                        ],
                        "constraints": [
                            "Do not suggest illegal or unethical plans or strategies.",
                            "Take reasonable budgetary limits into account.",
                        ],
                    },
                },
            },
        }
    ]

    system_prompt: str = UserConfigurable(
        default=(
            "Your job is to respond to a user-defined task, given in triple quotes, by "
            "invoking the `create_agent` function to generate an autonomous agent to "
            "complete the task. "
            "You should supply a role-based name for the agent (_GPT), "
            "an informative description for what the agent does, and 1 to 5 directives "
            "in each of the categories Best Practices and Constraints, "
            "that are optimally aligned with the successful completion "
            "of its assigned task.\n"
            "\n"
            "Example Input:\n"
            '"""Help me with marketing my business"""\n\n'
            "Example Call:\n"
            "```\n"
            f"{json.dumps(_example_call, indent=4)}"
            "\n```"
        )
    )
    user_prompt_template: str = UserConfigurable(default='"""{user_objective}"""')
    create_agent_function: dict = UserConfigurable(
        default=CompletionModelFunction(
            name="create_agent",
            description="Create a new autonomous AI agent to complete a given task.",
            parameters={
                "name": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    description="A short role-based name for an autonomous agent.",
                    required=True,
                ),
                "description": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    description=(
                        "An informative one sentence description "
                        "of what the AI agent does"
                    ),
                    required=True,
                ),
                "directives": JSONSchema(
                    type=JSONSchema.Type.OBJECT,
                    properties={
                        "best_practices": JSONSchema(
                            type=JSONSchema.Type.ARRAY,
                            minItems=1,
                            maxItems=5,
                            items=JSONSchema(
                                type=JSONSchema.Type.STRING,
                            ),
                            description=(
                                "One to five highly effective best practices "
                                "that are optimally aligned with the completion "
                                "of the given task"
                            ),
                            required=True,
                        ),
                        "constraints": JSONSchema(
                            type=JSONSchema.Type.ARRAY,
                            minItems=1,
                            maxItems=5,
                            items=JSONSchema(
                                type=JSONSchema.Type.STRING,
                            ),
                            description=(
                                "One to five reasonable and efficacious constraints "
                                "that are optimally aligned with the completion "
                                "of the given task"
                            ),
                            required=True,
                        ),
                    },
                    required=True,
                ),
            },
        ).schema
    )


class AgentProfileGenerator(PromptStrategy):
    default_configuration: AgentProfileGeneratorConfiguration = (
        AgentProfileGeneratorConfiguration()
    )

    def __init__(
        self,
        model_classification: LanguageModelClassification,
        system_prompt: str,
        user_prompt_template: str,
        create_agent_function: dict,
    ):
        self._model_classification = model_classification
        self._system_prompt_message = system_prompt
        self._user_prompt_template = user_prompt_template
        self._create_agent_function = CompletionModelFunction.parse(
            create_agent_function
        )

    @property
    def model_classification(self) -> LanguageModelClassification:
        return self._model_classification

    def build_prompt(self, user_objective: str = "", **kwargs) -> ChatPrompt:
        user_message = ChatMessage.user(
            self._user_prompt_template.format(
                user_objective=user_objective,
            )
        )
        prompt = ChatPrompt(
            messages=[user_message],
            functions=[self._create_agent_function],
        )
        return prompt

    def parse_response_content(
        self,
        arguments: dict,
    ) -> tuple[AIProfile, AIDirectives]:
        """Parse the actual text response from the objective model.

        Args:
            response_content: The raw response content from the objective model.

        Returns:
            The parsed response.

        """
        ai_profile = AIProfile(
            ai_name=arguments.get("name"),
            ai_role=arguments.get("description"),
        )
        ai_directives = AIDirectives(
            best_practices=arguments["directives"].get("best_practices"),
            constraints=arguments["directives"].get("constraints"),
            resources=[],
        )
        return ai_profile, ai_directives

class PropertyPrompter:
    def __init__(
            self, 
            profile_generator: AgentProfileGenerator,
            app_config: Config,
            provider: ChatModelProvider,
            base_prompt: str,
            task: str
        ):
        self.profile_generator = profile_generator
        self.app_config = app_config
        self.provider = provider
        self.base_prompt = base_prompt
        self.task = task

    async def prompt_for_property(
        self,
        property_prompt: str,
        first_line: bool = False
    ) -> str:
        ### Name
        prompt = self.profile_generator.build_prompt(
            f"{self.base_prompt}\n{property_prompt}\n{self.task}"
        )
        prop = (await self.provider.create_chat_completion(prompt.messages, model_name=self.app_config.smart_llm)).response['content']
        if prop is None:
            return self.prompt_for_property(property_prompt, first_line)
        if first_line:
            try:
                prop.index('\n') 
                prop = prop.split('\n')[0]
            except:
                prop = prop
        if prop is None:
            return self.prompt_for_property(property_prompt, first_line)
        return prop

async def generate_agent_profile_for_task(
    task: str,
    app_config: Config,
    llm_provider: ChatModelProvider,
) -> tuple[AIProfile, AIDirectives]:
    """Generates an AIConfig object from the given string.

    Returns:
    AIConfig: The AIConfig object tailored to the user's input
    """
    agent_profile_generator = AgentProfileGenerator(
        **AgentProfileGenerator.default_configuration.dict()  # HACK
    )
    
    base_prompt:str = (
        "Your job is to create an autonomous agent. Do not ask further questions, just reply with your answer. Do not add extra information, only reply with the information requested. Do not include code snippets in your reply."
    )

    prompter = PropertyPrompter(agent_profile_generator, app_config, llm_provider, base_prompt, task)

    name_prompt: str = "What should the name of the agent be for the instruction below? Only respond with the name of the agent, nothing else."
    desc_prompt: str = "Describe what the goal of the agent should be?"
    direct_prompt: str = "Create a bullet list of directives for the agent to follow."
    best_pract_prompt: str = f"{direct_prompt}\n What are the best practices that the agent should adhere to?"
    constraint_prompt: str = f"{direct_prompt}\n What are the constraints the agent should restrict itself from"
    profile: dict = {
        "name": await prompter.prompt_for_property(name_prompt, True),
        "description": await prompter.prompt_for_property(desc_prompt),
        "directives": {
            "best_practices": (await prompter.prompt_for_property(best_pract_prompt)).split('\n'),
            "constraints": (await prompter.prompt_for_property(constraint_prompt)).split('\n')
        }
    }

    logger.debug('Agent profile generated:')
    for key in profile.keys():
        logger.debug(f'{key}:\n{profile[key]}\n\n')

    # Debug LLM Output
    logger.debug(f"AI Config Generator Raw Output: {profile}")

    # Parse the output
    ai_profile, ai_directives = agent_profile_generator.parse_response_content(profile)

    return ai_profile, ai_directives
