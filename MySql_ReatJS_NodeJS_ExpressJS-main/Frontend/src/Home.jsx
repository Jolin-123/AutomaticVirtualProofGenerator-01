import React from 'react'
import './webpagestyle.css'
import { Link } from 'react-router-dom'
import mainPageSide02 from './images/mainPageSide02.png';
import folddingTT from './images/folddingTT.png';
import hangingT from './images/hangingT.png';
import hangingTgray from './images/hangingTgray.png';



function Home() {
  return (
    <div>
        {/* Home */}
             <div>
            <header> 
                <button className="menu"> &#9776;</button>
                <h1> 烘焙工坊</h1>
                <nav>
                    <ul>
                    <li><a href="/Home" target="_blank" > Home </a></li>
                        {/* <li><Link to="/Home">Home</Link></li> */}
                        <li><Link to="/NewProduct">Product</Link></li>
                        <li><Link to="/Contactus">Contact Us</Link></li>
                        <li><a href="/Login" target="_blank" >Login</a></li>

                    </ul>
                </nav>
            </header>  

            <main>
                <section className="news newsLayer">
                    <h2> LA Fashion ! </h2>
                    <p><h3> The rhythm of fashion, the dance of style, each piece tells a story of confidence.</h3></p>
                    {/* <p> <a href=".Contactus.html"> Learn More </a></p> */}
                    <p> <Link to="/Contactus"> Learn More </Link></p>
                </section>

{/*                 
                <section>
                    <div class="slider-container">
                        <div class="slider">
                            <div class="slide" style="background-color: red; color: white;">Slide 1</div>
                            <div class="slide" style="background-color: blue; color: white;">Slide 2</div>
                            <div class="slide" style="background-color: green; color: white;">Slide 3</div>
                            <div class="slide" style="background-color: yellow; color: black;">Slide 4</div>
                            <div class="slide" style="background-color: purple; color: white;">Slide 5</div>
                        </div>
                        <button class="prev" onclick="moveSlide(-1)">&#10094;</button>
                        <button class="next" onclick="moveSlide(1)">&#10095;</button>
                        <div class="dots">
                            <span class="dot" onclick="currentSlide(0)"> 1</span>
                            <span class="dot" onclick="currentSlide(1)"> 2</span>
                            <span class="dot" onclick="currentSlide(2)"> 3</span>
                            <span class="dot" onclick="currentSlide(3)"> 4</span>
                            <span class="dot" onclick="currentSlide(4)"> 5</span>
                        </div>
                    </div>
                </section> */}


                <section className="slogan">
                    <blockquote> "Carefully selected premium materials, crafted with ingenuity, allowing fashion to flow through every T-shirt."</blockquote>
                </section>

                <section className="shop">
                     <img src={mainPageSide02 } alt="mainPageSide02 " />
                    {/* <img src="./images/mainPageSide02.png" alt = "烘焙工坊大安店"/> */}
                    <div className="info">
                    <p> <h2>  T-Shirt Manufacturing Company </h2>
                        Grand Opening of our new online store on July 15th <br/> 
                            Enjoy the pleasure of online shopping, fashion by your side. 
                            Starting today, join our membership group when shopping and enjoy exclusive discounts, 
                            looking forward to your participation! </p>
                    </div>
                </section>

                <section className="product">
                    <div>
                        {/* <img src="./images/lacity03.png" alt = "Long-Sleeve T-Shirts"/> */}
                        <img src={hangingT } alt="hangingT " />
                        <h3>   Casual Simple Differnt Style T-Shirts</h3>
                        <p>  Classic style, comfortable fit  <br/>  Letting you experience the perfect blend of vintage and trendy </p>
                    </div>
                    <div>
                        {/* <img src="./images/folddingTT.png" alt="Long-Sleeve T-Shirts "/> */}
                        <img src={folddingTT } alt="folddingTT " />
                        <h3>  High Quality Fabric, Classic Design </h3>
                        <p>  Simple and generous, casual and comfortable.  <br/> The perfect choice for your daily life </p>
                    </div>
                    <div>
                        {/* <img src="./images/hangingTgray.png" alt=" Unique Custom T-Shirts"/> */}
                        <img src={hangingTgray } alt="hangingTgray " />
                        <h3>  Unique Custom T-Shirts</h3>
                        <p>  Customized to be different from the crowd <br/> Allowing you to have a unique fashion experience </p>
                    </div>
                </section>

                <section className="newsletter">
                    <p>  Subscribe to our fashion newsletter, regularly receive the latest styles and special offers! </p>
                    <form>
                        <input type="email" placeholder="Please enter your Email address " required/>
                        <button type="submit"> Submit</button>
                    </form>
                </section>
            </main>

            <footer>
                <p> Copyright @ 2023 Fashion T-Shirt Manufacturing Company All Right Reserved </p>
            </footer>
        </div>

    </div>
  )
}

export default Home