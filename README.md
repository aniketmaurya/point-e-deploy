# Lightning Dream Generator

<p align="center">
  <img width="250" alt="logo" src="https://i.ibb.co/BrsfpBj/Visualize-Your-word-banner-small.jpg"/>
  <br>
  <strong>Create beautiful works of art in seconds with the Lightning Dream Generator App.</strong>
  <br>
  <p align="center" style="color:grey">Powered by <a href="https://stability.ai/blog/stable-diffusion-public-release">Stable Diffusion</a></p>
</p>
<p align="center">
</p>

______________________________________________________________________

With this you can launch a React UI for generating art from texts, integrate a Slack Command Bot that can generate art
in your slack workspace.

# Getting started

## Launch React UI

![Demo image](./assets/demo.png)

```bash
lightning install app https://github.com/Lightning-AI/LAI-Stable-Diffusion-App

cd LAI-Stable-Diffusion-App

lightning run app app.py --cloud --env access_token=HUGGINGFACE_TOKEN
```

<sup>Learn more about access token <a href="https://huggingface.co/docs/hub/security-tokens">here</a>.</sup>

This will launch your app on the Lightning.ai cloud. You can enter the text prompt in the input bar and click on the
`Dream it` button to generate art.

## Integrate with Slack Bot

You can integrate this app in your Slack Workspace and send art in slack channels.

This app uses the [Slack Command Bot Component](https://github.com/Lightning-AI/LAI-slack-command-bot-Component) for to
interact with Slack command.

[![Watch the video](https://img.youtube.com/vi/KfQcXzWFR9I/default.jpg)](https://youtu.be/KfQcXzWFR9I)

### Steps to create the Slack Command Bot

**Step 1:**
Goto https://api.slack.com and create an app.

**Step 2:**
Copy the following tokens and secrets from the Slack API settings by going to https://api.slack.com/apps. These tokens
have to be passed either as argument or environment variable to [SlackCommandBot](https://github.com/Lightning-AI/LAI-slack-command-bot-Component/blob/main/slack_command_bot/component.py#L18) class.

<details>
  <summary>Required Token name and environment variables: </summary>

- Client ID (SLACK_CLIENT_ID)
- Client Secret (CLIENT_SECRET)
- Signing Secret (SIGNING_SECRET)
- Bot User OAuth Token (BOT_TOKEN)
- App-Level Token (SLACK_TOKEN)

</details>

**Step 3:**

Implement the `SlackCommandBot.handle_command(...)` method the way you want to interact with the commands.
The return value will be shown only to you.

> ![](./assets/slack-ss.png)

**Step 4:** (optional)

If you want your slack app to be distributable to public then you need to
implement `SlackCommandBot.save_new_workspace(...)` which should save `team_id` and its corresponding `bot_token` into a
database.

During the `handle_command(...)` method you will need to fetch `bot_token` based on the received `team_id`.
